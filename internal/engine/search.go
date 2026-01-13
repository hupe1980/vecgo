package engine

import (
	"context"
	"fmt"
	"iter"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/model"
)

type segBitmap struct {
	seg    segment.Segment
	bitmap segment.Bitmap
	ts     segment.Filter // Tombstone filter
}

var segBitmapPool = sync.Pool{
	New: func() any {
		s := make([]segBitmap, 0, 16)
		return &s
	},
}

// SearchIter performs a k-NN search and yields results via an iterator.
// This allows for zero-copy streaming of results.
//
// New in v1.0: Returns iter.Seq2 for high-performance streaming.
func (e *Engine) SearchIter(ctx context.Context, q []float32, k int, opts ...func(*model.SearchOptions)) iter.Seq2[model.Candidate, error] {
	return func(yield func(model.Candidate, error) bool) {
		start := time.Now()
		var count int
		var err error
		defer func() {
			// Heuristic: determine segment type.
			e.metrics.OnSearch(time.Since(start), "mixed", k, count, err)
		}()

		select {
		case <-e.closeCh:
			yield(model.Candidate{}, ErrClosed)
			return
		default:
		}

		if err = e.validateVector(q); err != nil {
			yield(model.Candidate{}, err)
			return
		}
		if k <= 0 {
			yield(model.Candidate{}, fmt.Errorf("%w: k must be > 0", ErrInvalidArgument))
			return
		}

		// Parse options - default: include metadata and payload (best DX)
		// Use WithoutData() to disable for high-throughput scenarios
		options := model.SearchOptions{
			K:               k,
			RefineFactor:    1.0,
			IncludeMetadata: true,
			IncludePayload:  true,
		}
		for _, opt := range opts {
			opt(&options)
		}

		// Acquire Snapshot
		snap, loadErr := e.loadSnapshot()
		if loadErr != nil {
			err = loadErr
			yield(model.Candidate{}, err)
			return
		}
		defer snap.DecRef()

		// Acquire Searcher
		s := searcher.Get()
		defer searcher.Put(s)

		// Normalize cosine queries
		qExec := q
		if e.metric == distance.MetricCosine {
			if cap(s.ScratchVec) < e.dim {
				s.ScratchVec = make([]float32, e.dim)
			}
			s.ScratchVec = s.ScratchVec[:e.dim]
			copy(s.ScratchVec, q)
			if !distance.NormalizeL2InPlace(s.ScratchVec) {
				err = fmt.Errorf("%w: cannot normalize zero query", ErrInvalidArgument)
				yield(model.Candidate{}, err)
				return
			}
			qExec = s.ScratchVec
		}

		// Initialize Global Heap
		descending := e.metric != distance.MetricL2
		s.Heap.Reset(descending)

		// 1. Gather candidates from all segments
		searchK := int(float32(k) * options.RefineFactor)
		if searchK < k {
			searchK = k
		}

		currentLSN := e.lsn.Load()

		e.tombstonesMu.RLock()
		var activeFilter segment.Filter
		// In read-only mode, there's no active memtable
		if snap.active != nil {
			if vt, ok := e.tombstones[snap.active.ID()]; ok {
				activeFilter = e.acquireTombstoneFilter(vt, currentLSN)
			}
		}
		defer func() { e.releaseTombstoneFilter(activeFilter) }()

		// Fast path: L0-only search (no immutable segments)
		// Skip goroutine/channel overhead when only MemTable exists
		// In read-only mode (snap.active == nil), we skip this path
		if len(snap.sortedSegments) == 0 && snap.active != nil {
			e.tombstonesMu.RUnlock()

			if err := snap.active.Search(ctx, qExec, searchK, activeFilter, options, s); err != nil {
				yield(model.Candidate{}, err)
				return
			}
		} else if len(snap.sortedSegments) == 0 && snap.active == nil {
			// Read-only mode with no segments (empty database)
			e.tombstonesMu.RUnlock()
			// Return empty results - nothing to search
		} else {
			// Create filters for segments while holding the lock
			// OPTIMIZATION: Use pooled segment filters slice to avoid allocation
			if cap(s.SegmentFilters) < len(snap.sortedSegments) {
				s.SegmentFilters = make([]any, len(snap.sortedSegments))
			}
			s.SegmentFilters = s.SegmentFilters[:len(snap.sortedSegments)]

			for i, seg := range snap.sortedSegments {
				if vt, ok := e.tombstones[seg.ID()]; ok {
					s.SegmentFilters[i] = e.acquireTombstoneFilter(vt, currentLSN)
				} else {
					s.SegmentFilters[i] = nil
				}
			}
			e.tombstonesMu.RUnlock()

			defer func() {
				for _, f := range s.SegmentFilters {
					if f != nil {
						e.releaseTombstoneFilter(f.(segment.Filter))
					}
				}
			}()

			// Strategy Selection: Check for Bitmap Pre-Filtering
			var usedBitmapStrategy bool

			// Adaptive threshold based on k and total vector count.
			// For small k and highly selective filters, brute-force is faster.
			// For large k or low selectivity, HNSW is better.
			// Heuristic: threshold = max(k * 50, 500) capped at 5000
			adaptiveThreshold := uint64(k * 50)
			if adaptiveThreshold < 500 {
				adaptiveThreshold = 500
			}
			if adaptiveThreshold > 5000 {
				adaptiveThreshold = 5000
			}

			if options.Filter != nil {
				// Try to get bitmaps from all segments

				// We need bitmaps for Active + Immutable
				bitmapsPtr := segBitmapPool.Get().(*[]segBitmap)
				bitmaps := (*bitmapsPtr)[:0]
				defer func() {
					// Avoid retaining references to segments/bitmaps/filters across searches.
					for i := range bitmaps {
						bitmaps[i] = segBitmap{}
					}
					*bitmapsPtr = bitmaps[:0]
					segBitmapPool.Put(bitmapsPtr)
				}()

				totalHits := uint64(0)
				possible := true

				// Active Segment (only if not in read-only mode)
				if snap.active != nil {
					if b, err := snap.active.EvaluateFilter(ctx, options.Filter); err == nil && b != nil {
						hits := b.Cardinality()
						totalHits += hits
						bitmaps = append(bitmaps, segBitmap{seg: snap.active, bitmap: b, ts: activeFilter})
					} else if err != nil {
						possible = false
					} else {
						// b == nil means match ALL. This is bad for pre-filtering (implies full scan).
						// Fallback to HNSW if matches all (likely high selectivity).
						possible = false
					}
				}

				// Immutable Segments
				if possible {
					for i, seg := range snap.sortedSegments {
						b, err := seg.EvaluateFilter(ctx, options.Filter)
						if err == nil && b != nil {
							hits := b.Cardinality()
							totalHits += hits
							if totalHits > adaptiveThreshold {
								possible = false
								break
							}
							var ts segment.Filter
							if f := s.SegmentFilters[i]; f != nil {
								ts = f.(segment.Filter)
							}
							bitmaps = append(bitmaps, segBitmap{seg: seg, bitmap: b, ts: ts})
						} else {
							// Error or MatchesAll
							possible = false
							break
						}
					}
				}

				if possible && totalHits <= adaptiveThreshold {
					usedBitmapStrategy = true

					// Execute Bitmap Brute Force
					var scoreFunc func([]float32, []float32) float32
					if e.metric == distance.MetricL2 {
						scoreFunc = distance.SquaredL2
					} else {
						scoreFunc = distance.Dot
					}

					for _, sb := range bitmaps {
						// Collect RowIDs
						rowIDs := s.ScratchIDs[:0]
						needed := int(sb.bitmap.Cardinality())
						if cap(rowIDs) < needed {
							rowIDs = make([]uint32, 0, needed)
						}

						sb.bitmap.ForEach(func(id uint32) bool {
							// Apply Tombstone Filter
							if sb.ts != nil && !sb.ts.Matches(id) {
								return true // Skip deleted
							}
							rowIDs = append(rowIDs, id)
							return true
						})

						if len(rowIDs) == 0 {
							continue
						}

						// Fetch Vectors
						batch, err := sb.seg.Fetch(ctx, rowIDs, []string{"vector"})
						if err != nil {
							// If fetch fails, we can't search this segment.
							// Fail query? Or skip?
							yield(model.Candidate{}, err)
							return
						}

						// Compute Scores
						for i := 0; i < batch.RowCount(); i++ {
							vec := batch.Vector(i)
							if vec == nil {
								continue
							}

							dist := scoreFunc(qExec, vec)

							c := searcher.InternalCandidate{
								SegmentID: uint32(sb.seg.ID()),
								RowID:     rowIDs[i],
								Score:     dist,
								Approx:    false, // Exact
							}

							// Push to heap
							if s.Heap.Len() < searchK {
								s.Heap.Push(c)
							} else {
								worst := s.Heap.Peek()
								if searcher.InternalCandidateBetter(c, worst, descending) {
									s.Heap.Pop()
									s.Heap.Push(c)
								}
							}
						}

						s.ScratchIDs = rowIDs[:0]
					}
				}
			}

			if !usedBitmapStrategy {
				// Parallel Segment Search (scatter-gather pattern)
				// Revised to avoid errgroup overhead
				numSegments := len(snap.sortedSegments)
				if snap.active != nil {
					numSegments++ // Include memtable
				}

				concurrency := runtime.GOMAXPROCS(0)
				if concurrency > numSegments {
					concurrency = numSegments
				}
				if concurrency == 0 {
					concurrency = 1
				}

				var wg sync.WaitGroup
				sem := make(chan struct{}, concurrency)

				// Storage for results to avoid locking the main heap
				// OPTIMIZATION: Use pooled result buffers to avoid allocation
				maxRes := numSegments * searchK
				if cap(s.ParallelResults) < maxRes {
					s.ParallelResults = make([]searcher.InternalCandidate, maxRes)
				}

				if cap(s.ParallelSlices) < numSegments {
					s.ParallelSlices = make([][]searcher.InternalCandidate, numSegments)
				}
				s.ParallelSlices = s.ParallelSlices[:numSegments]
				// Ensure clean state (nil slices)
				for i := range s.ParallelSlices {
					s.ParallelSlices[i] = nil
				}
				results := s.ParallelSlices
				var firstErr error
				var firstErrOnce sync.Once

				runSearch := func(idx int, seg segment.Segment, filt segment.Filter) {
					defer wg.Done()

					// Acquire semaphore
					sem <- struct{}{}
					defer func() { <-sem }()

					// Acquire local searcher
					localS := searcher.Get()
					defer searcher.Put(localS)
					localS.Heap.Reset(descending)

					if err := seg.Search(ctx, qExec, searchK, filt, options, localS); err != nil {
						firstErrOnce.Do(func() { firstErr = err })
						return
					}

					// Extract results with ZERO heap allocation per segment
					if localS.Heap.Len() > 0 {
						// Calculate slice of persistent buffer to use
						start := idx * searchK
						// Note: s.ParallelResults cap = numSegments * searchK.
						// Safely slice:
						dest := s.ParallelResults[start : start : start+searchK] // len=0, cap=searchK

						for localS.Heap.Len() > 0 {
							dest = append(dest, localS.Heap.Pop())
						}
						results[idx] = dest
					}
				}

				// 1a. Search MemTable (only if not in read-only mode)
				if snap.active != nil {
					wg.Add(1)
					go runSearch(0, snap.active, activeFilter)
				}

				// 1b. Search Segments
				for i, seg := range snap.sortedSegments {
					wg.Add(1)
					// In read-only mode, index starts at 0 instead of 1
					idx := i + 1
					if snap.active == nil {
						idx = i
					}
					var f segment.Filter
					if v := s.SegmentFilters[i]; v != nil {
						f = v.(segment.Filter)
					}
					go runSearch(idx, seg, f)
				}

				wg.Wait()

				if firstErr != nil {
					yield(model.Candidate{}, firstErr)
					return
				}

				// Merge results
				for _, candidates := range results {
					for _, c := range candidates {
						if s.Heap.Len() < searchK {
							s.Heap.Push(c)
						} else {
							worst := s.Heap.Peek()
							if searcher.InternalCandidateBetter(c, worst, descending) {
								s.Heap.Pop()
								s.Heap.Push(c)
							}
						}
					}
				}
			}
		}

		// 2. Extract candidates for reranking
		s.CandidateBuffer = s.CandidateBuffer[:0]
		for s.Heap.Len() > 0 {
			s.CandidateBuffer = append(s.CandidateBuffer, s.Heap.Pop())
		}

		// 3. Rerank
		slices.SortFunc(s.CandidateBuffer, func(a, b searcher.InternalCandidate) int {
			if a.SegmentID < b.SegmentID {
				return -1
			}
			if a.SegmentID > b.SegmentID {
				return 1
			}
			return 0
		})

		s.Results = s.Results[:0]

		for i := 0; i < len(s.CandidateBuffer); {
			segID := s.CandidateBuffer[i].SegmentID
			j := i + 1
			for j < len(s.CandidateBuffer) && s.CandidateBuffer[j].SegmentID == segID {
				j++
			}

			// Batch for segID is [i, j)
			var seg segment.Segment
			if snap.active != nil && segID == uint32(snap.active.ID()) {
				seg = snap.active
			} else {
				if s, ok := snap.segments[model.SegmentID(segID)]; ok {
					seg = s
				}
			}

			if seg != nil {
				// Convert InternalCandidate batch to model.Candidate batch for Rerank interface
				chunk := s.CandidateBuffer[i:j]

				// Ensure ModelScratch has enough capacity
				if cap(s.ModelScratch) < len(chunk) {
					s.ModelScratch = make([]model.Candidate, len(chunk))
				}
				s.ModelScratch = s.ModelScratch[:len(chunk)]

				for k, c := range chunk {
					s.ModelScratch[k] = c.ToModel()
				}

				var rErr error
				s.Results, rErr = seg.Rerank(ctx, qExec, s.ModelScratch, s.Results)
				if rErr != nil {
					err = rErr
					yield(model.Candidate{}, err)
					return
				}
			}
			i = j
		}

		// 4. Final Top-K Selection
		s.Heap.Reset(descending)
		for _, c := range s.Results {
			// Convert model.Candidate back to InternalCandidate for Heap
			ic := searcher.FromModel(c)
			if s.Heap.Len() < k {
				s.Heap.Push(ic)
			} else {
				top := s.Heap.Candidates[0]
				if searcher.InternalCandidateBetter(ic, top, descending) {
					s.Heap.ReplaceTop(ic)
				}
			}
		}

		// 5. Extract Final Results
		s.Results = s.Results[:0]
		for s.Heap.Len() > 0 {
			s.Results = append(s.Results, s.Heap.Pop().ToModel())
		}
		slices.Reverse(s.Results)

		// 6. Fetch PKs and Materialize Data
		var cols []string
		if options.IncludeVector {
			cols = append(cols, "vector")
		}
		if options.IncludeMetadata {
			cols = append(cols, "metadata")
		}
		if options.IncludePayload {
			cols = append(cols, "payload")
		}

		slices.SortFunc(s.Results, func(a, b model.Candidate) int {
			if a.Loc.SegmentID < b.Loc.SegmentID {
				return -1
			}
			if a.Loc.SegmentID > b.Loc.SegmentID {
				return 1
			}
			return 0
		})

		for i := 0; i < len(s.Results); {
			segID := s.Results[i].Loc.SegmentID
			j := i + 1
			for j < len(s.Results) && s.Results[j].Loc.SegmentID == segID {
				j++
			}

			var seg segment.Segment
			if snap.active != nil && segID == snap.active.ID() {
				seg = snap.active
			} else {
				if s, ok := snap.segments[segID]; ok {
					seg = s
				}
			}

			if seg != nil {
				countBatch := j - i
				if cap(s.ScratchIDs) < countBatch {
					s.ScratchIDs = make([]uint32, countBatch)
				}
				s.ScratchIDs = s.ScratchIDs[:countBatch]

				for k := 0; k < countBatch; k++ {
					s.ScratchIDs[k] = uint32(s.Results[i+k].Loc.RowID)
				}

				if len(cols) == 0 {
					if cap(s.ScratchForeignIDs) < countBatch {
						s.ScratchForeignIDs = make([]model.ID, countBatch)
					}
					s.ScratchForeignIDs = s.ScratchForeignIDs[:countBatch]

					if err = seg.FetchIDs(ctx, s.ScratchIDs, s.ScratchForeignIDs); err != nil {
						yield(model.Candidate{}, err)
						return
					}
					for k := 0; k < countBatch; k++ {
						s.Results[i+k].ID = s.ScratchForeignIDs[k]
					}
				} else {
					batch, fErr := seg.Fetch(ctx, s.ScratchIDs, cols)
					if fErr != nil {
						err = fErr
						yield(model.Candidate{}, err)
						return
					}

					for k := 0; k < countBatch; k++ {
						s.Results[i+k].ID = batch.ID(k)
						if options.IncludeVector {
							s.Results[i+k].Vector = batch.Vector(k)
						}
						if options.IncludeMetadata {
							s.Results[i+k].Metadata = batch.Metadata(k)
						}
						if options.IncludePayload {
							s.Results[i+k].Payload = batch.Payload(k)
						}
					}
				}
			}
			i = j
		}

		// 7. Restore Order (Best Score First)
		slices.SortFunc(s.Results, func(a, b model.Candidate) int {
			if descending {
				if a.Score > b.Score {
					return -1
				}
				if a.Score < b.Score {
					return 1
				}
			} else {
				if a.Score < b.Score {
					return -1
				}
				if a.Score > b.Score {
					return 1
				}
			}
			if a.Loc.SegmentID < b.Loc.SegmentID {
				return -1
			}
			if a.Loc.SegmentID > b.Loc.SegmentID {
				return 1
			}
			if a.Loc.RowID < b.Loc.RowID {
				return -1
			}
			if a.Loc.RowID > b.Loc.RowID {
				return 1
			}
			return 0
		})

		// 8. Visibility Check
		validCount := 0
		for _, cand := range s.Results {
			loc, ok := e.pkIndex.Get(cand.ID, currentLSN)
			if !ok {
				continue
			}
			if loc != cand.Loc {
				continue
			}
			s.Results[validCount] = cand
			validCount++
		}
		s.Results = s.Results[:validCount]

		count = len(s.Results)
		for _, cand := range s.Results {
			if !yield(cand, nil) {
				return
			}
		}
	}
}

// Search performs a k-NN search and returns a slice of candidates.
// This is a convenience wrapper around SearchIter.
func (e *Engine) Search(ctx context.Context, q []float32, k int, opts ...func(*model.SearchOptions)) ([]model.Candidate, error) {
	// Pre-allocate slice?
	// We don't know the exact count if error occurs, but k is upper bound.
	res := make([]model.Candidate, 0, k)

	next := e.SearchIter(ctx, q, k, opts...)
	for c, err := range next {
		if err != nil {
			return nil, err
		}
		res = append(res, c)
	}
	return res, nil
}
