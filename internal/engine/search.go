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
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// segFilter holds filter results for a segment.
// Uses FilterResult for zero-alloc hot path.
type segFilter struct {
	seg segment.Segment
	fr  imetadata.FilterResult // Zero-alloc filter result
	ts  segment.Filter         // Tombstone filter
}

var segFilterPool = sync.Pool{
	New: func() any {
		s := make([]segFilter, 0, 16)
		return &s
	},
}

// rowIDBufferPool provides reusable buffers for collecting row IDs across segments.
// This avoids per-search allocations when using CloneInto.
var rowIDBufferPool = sync.Pool{
	New: func() any {
		// Start with 4KB capacity (1024 uint32s) - grows as needed
		buf := make([]uint32, 0, 1024)
		return &buf
	},
}

// Package-level comparison functions to avoid closure allocation in hot paths.
// These are used with slices.SortFunc to eliminate per-call closure overhead.

// cmpInternalCandidateBySegment compares InternalCandidates by SegmentID.
func cmpInternalCandidateBySegment(a, b searcher.InternalCandidate) int {
	if a.SegmentID < b.SegmentID {
		return -1
	}
	if a.SegmentID > b.SegmentID {
		return 1
	}
	return 0
}

// cmpCandidateBySegment compares Candidates by SegmentID.
func cmpCandidateBySegment(a, b model.Candidate) int {
	if a.Loc.SegmentID < b.Loc.SegmentID {
		return -1
	}
	if a.Loc.SegmentID > b.Loc.SegmentID {
		return 1
	}
	return 0
}

// cmpCandidateByScoreAsc compares Candidates by Score ascending (best = lowest).
func cmpCandidateByScoreAsc(a, b model.Candidate) int {
	if a.Score < b.Score {
		return -1
	}
	if a.Score > b.Score {
		return 1
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
}

// cmpCandidateByScoreDesc compares Candidates by Score descending (best = highest).
func cmpCandidateByScoreDesc(a, b model.Candidate) int {
	if a.Score > b.Score {
		return -1
	}
	if a.Score < b.Score {
		return 1
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
		searchK := max(int(float32(k)*options.RefineFactor), k)

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

			// CURSOR-BASED SEARCH: Zero-allocation filtered search
			// Uses push-based iteration instead of materializing bitmaps
			if options.Filter != nil {
				selectivityCutoff := 0.30
				if options.SelectivityCutoff > 0 {
					selectivityCutoff = options.SelectivityCutoff
				}
				forcePreFilter := options.PreFilter != nil && *options.PreFilter

				// Get cursor from memtable (zero-alloc)
				cursor := snap.active.FilterCursor(options.Filter)

				if !cursor.IsAll() && !cursor.IsEmpty() {
					segSel := float64(cursor.EstimateCardinality()) / float64(snap.active.RowCount())
					if forcePreFilter || segSel <= selectivityCutoff {
						// Use cursor-based streaming execution
						config := CursorSearchConfig{
							SelectivityCutoff: selectivityCutoff,
							BatchSize:         64,
							ForcePreFilter:    forcePreFilter,
						}

						used, err := e.searchSegmentWithCursor(
							ctx, snap.active, cursor, activeFilter,
							config, s, qExec, searchK, descending,
						)
						if err != nil {
							yield(model.Candidate{}, err)
							return
						}
						if used {
							// Skip HNSW search since we used streaming
							goto rerank
						}
					}
				}
			}

			// Fall back to HNSW search for high selectivity or no filter
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

			// Track if high selectivity detected (skip bitmap, use HNSW with post-filter)
			var skipBitmapForHNSW bool

			// Track maximum selectivity across segments for HNSW adaptive traversal
			var maxSelectivity float64

			// Selectivity-based cutoff: above this threshold, bitmap overhead dominates
			// and HNSW with post-filtering is faster.
			// Default: 30% (0.30) - based on benchmark analysis showing bitmap overhead
			// dominates above ~30% selectivity. User can override via SelectivityCutoff.
			selectivityCutoff := 0.30
			if options.SelectivityCutoff > 0 {
				selectivityCutoff = options.SelectivityCutoff
			}

			// Adaptive threshold based on k and total vector count (legacy fallback).
			// For small k and highly selective filters, brute-force is faster.
			// Heuristic: threshold = max(k * 50, 500) capped at 5000
			adaptiveThreshold := min(max(uint64(k*50), 500), 5000)

			// User can force pre-filtering via WithPreFilter(true) to bypass adaptive heuristic
			forcePreFilter := options.PreFilter != nil && *options.PreFilter

			if options.Filter != nil {
				// STREAMING EXECUTION: Process each segment immediately
				// This avoids global FilterResult collection and reduces memory churn.
				// Key insight: FilterResults should die at segment boundary.

				// Cursor config for consistent settings
				cursorConfig := CursorSearchConfig{
					SelectivityCutoff: selectivityCutoff,
					BatchSize:         64,
					ForcePreFilter:    forcePreFilter,
				}

				// Track if we used streaming strategy
				usedStreaming := false

				// For immutable segments, we still need QueryScratch
				qs := imetadata.GetQueryScratch()
				defer imetadata.PutQueryScratch(qs)

				// Process Active Segment (memtable) with cursor
				if snap.active != nil {
					cursor := snap.active.FilterCursor(options.Filter)
					if !cursor.IsAll() && !cursor.IsEmpty() {
						segSel := float64(cursor.EstimateCardinality()) / float64(snap.active.RowCount())
						if segSel > maxSelectivity {
							maxSelectivity = segSel
						}
						if forcePreFilter || segSel <= selectivityCutoff {
							used, err := e.searchSegmentWithCursor(
								ctx, snap.active, cursor, activeFilter,
								cursorConfig, s, qExec, searchK, descending,
							)
							if err != nil {
								yield(model.Candidate{}, err)
								return
							}
							if used {
								usedStreaming = true
							}
						} else {
							// High selectivity in memtable - prefer HNSW
							skipBitmapForHNSW = true
						}
					}
				}

				// Process Immutable Segments using FilterCursor when available
				// This eliminates Roaring bitmap allocations in the hot path
				if !skipBitmapForHNSW {
					for i, seg := range snap.sortedSegments {
						// SEGMENT PRUNING
						if e.canPruneSegment(seg.ID(), options.Filter) {
							continue
						}

						// Try cursor-based path first (zero-alloc)
						// Note: seg.Segment accesses the embedded interface
						if fc, ok := seg.Segment.(SegmentWithFilterCursor); ok {
							cursor := fc.FilterCursor(options.Filter)
							if cursor.IsAll() {
								// Matches all - use HNSW for this segment
								// But only if this segment has a graph index
								if segmentHasGraphIndex(seg) {
									skipBitmapForHNSW = true
									maxSelectivity = 1.0
									break
								}
								// Flat segment: can't use HNSW, fall through to brute-force with cursor
							}

							if !cursor.IsEmpty() {
								segSel := float64(cursor.EstimateCardinality()) / float64(seg.RowCount())
								if segSel > maxSelectivity {
									maxSelectivity = segSel
								}
								// High selectivity routing:
								// - For graph segments (HNSW/DiskANN): use graph traversal
								// - For flat segments: ALWAYS use cursor (avoids expensive MatchesBinary)
								if !forcePreFilter && segSel > selectivityCutoff && segmentHasGraphIndex(seg) {
									// Graph segment with high selectivity - switch to HNSW for all remaining
									skipBitmapForHNSW = true
									break
								}

								var ts segment.Filter
								if f := s.SegmentFilters[i]; f != nil {
									ts = f.(segment.Filter)
								}

								used, err := e.searchSegmentWithCursor(
									ctx, seg, cursor, ts,
									cursorConfig, s, qExec, searchK, descending,
								)
								if err != nil {
									yield(model.Candidate{}, err)
									return
								}
								if used {
									usedStreaming = true
								}
							}
							continue
						}

						// Fallback: use FilterResult path (legacy)
						fr, err := e.evaluateSegmentFilterResult(ctx, seg, options.Filter, qs)
						if err != nil {
							// Error - fallback to HNSW for remaining
							break
						}

						if fr.IsAll() {
							// Matches all - use HNSW for this segment (if it has a graph)
							if segmentHasGraphIndex(seg) {
								skipBitmapForHNSW = true
								maxSelectivity = 1.0
								break
							}
							// Flat segment: can't use HNSW, fall through to brute-force with cursor
						}

						if !fr.IsEmpty() {
							segSel := float64(fr.Cardinality()) / float64(seg.RowCount())
							if segSel > maxSelectivity {
								maxSelectivity = segSel
							}
							// High selectivity routing:
							// - For graph segments (HNSW/DiskANN): use graph traversal
							// - For flat segments: ALWAYS use cursor (avoids expensive MatchesBinary)
							if !forcePreFilter && segSel > selectivityCutoff && segmentHasGraphIndex(seg) {
								// Graph segment with high selectivity - switch to HNSW for all remaining
								skipBitmapForHNSW = true
								break
							}

							var ts segment.Filter
							if f := s.SegmentFilters[i]; f != nil {
								ts = f.(segment.Filter)
							}

							// Use cursor wrapper for FilterResult
							cursor := imetadata.NewFilterResultCursor(fr)
							used, err := e.searchSegmentWithCursor(
								ctx, seg, cursor, ts,
								cursorConfig, s, qExec, searchK, descending,
							)
							if err != nil {
								yield(model.Candidate{}, err)
								return
							}
							if used {
								usedStreaming = true
							}
						}
					}
				}

				// If we used streaming, we're done with brute-force path
				if usedStreaming {
					usedBitmapStrategy = true
				}
				// If high selectivity detected, we skip bitmap paths but still run HNSW
				// (usedBitmapStrategy stays false, so HNSW will run with post-filter)
			}

			// Legacy path (kept for fallback - will be removed after validation)
			// Skip if: streaming was used, OR high selectivity was detected
			if options.Filter != nil && !usedBitmapStrategy && !skipBitmapForHNSW {
				// Zero-alloc filter path using FilterResult
				// Get query scratch for filter evaluation
				qs := imetadata.GetQueryScratch()
				defer imetadata.PutQueryScratch(qs)

				// Collect filter results for each segment
				filtersPtr := segFilterPool.Get().(*[]segFilter)
				filters := (*filtersPtr)[:0]

				// Get pooled buffer for row ID collection (avoids per-Clone allocations)
				rowBufPtr := rowIDBufferPool.Get().(*[]uint32)
				rowBuf := (*rowBufPtr)[:0]

				defer func() {
					// Clear references (no pool return needed for FilterResult - query-scoped)
					for i := range filters {
						filters[i] = segFilter{}
					}
					*filtersPtr = filters[:0]
					segFilterPool.Put(filtersPtr)

					// Return row buffer to pool
					*rowBufPtr = rowBuf[:0]
					rowIDBufferPool.Put(rowBufPtr)
				}()

				totalHits := uint64(0)
				possible := true

				// Active Segment (only if not in read-only mode)
				if snap.active != nil {
					// Use zero-alloc EvaluateFilterResult for memtable
					fr, err := snap.active.EvaluateFilterResult(ctx, options.Filter, qs)
					if err != nil {
						possible = false
					} else if fr.IsAll() {
						// FilterAll = matches all = fallback to HNSW for this segment
						possible = false
					} else if !fr.IsEmpty() {
						hits := uint64(fr.Cardinality())
						totalHits += hits

						// SELECTIVITY-BASED CUTOFF: Check memtable selectivity
						memtableSelectivity := float64(hits) / float64(snap.active.RowCount())
						if !forcePreFilter && memtableSelectivity > selectivityCutoff {
							// High selectivity - bail out to HNSW
							possible = false
						} else {
							// CloneInto to prevent aliasing when qs.TmpRowIDs is reused
							// Uses pooled buffer to avoid per-segment allocations
							var cloned imetadata.FilterResult
							cloned, rowBuf = fr.CloneInto(rowBuf)
							filters = append(filters, segFilter{seg: snap.active, fr: cloned, ts: activeFilter})
						}
					}
					// else: fr.IsEmpty() = no matches in this segment = skip (valid)
				}

				// Immutable Segments - use EvaluateFilterResult where available
				// With segment pruning: skip segments that can't possibly match
				if possible {
					for i, seg := range snap.sortedSegments {
						// SEGMENT PRUNING: Skip segments that can be pruned based on manifest stats
						// This is O(1) per segment and avoids opening/reading segments entirely
						if e.canPruneSegment(seg.ID(), options.Filter) {
							// Segment can be skipped - no possible matches based on stats
							continue
						}

						// Try zero-alloc path first
						fr, err := e.evaluateSegmentFilterResult(ctx, seg, options.Filter, qs)
						if err != nil {
							// Error - fallback to HNSW
							possible = false
							break
						} else if fr.IsAll() {
							// FilterAll = matches all = fallback to HNSW for this segment
							possible = false
							break
						} else if !fr.IsEmpty() {
							hits := uint64(fr.Cardinality())
							totalHits += hits

							// SELECTIVITY-BASED CUTOFF: Check segment selectivity
							// If segment selectivity > cutoff, bitmap overhead dominates
							// and HNSW with post-filtering is faster for this segment.
							segmentSelectivity := float64(hits) / float64(seg.RowCount())
							if !forcePreFilter && segmentSelectivity > selectivityCutoff {
								// High selectivity segment detected - bail out to HNSW
								possible = false
								break
							}

							// Also check absolute threshold (legacy fallback)
							if !forcePreFilter && totalHits > adaptiveThreshold {
								possible = false
								break
							}
							var ts segment.Filter
							if f := s.SegmentFilters[i]; f != nil {
								ts = f.(segment.Filter)
							}
							// CloneInto to prevent aliasing when qs.TmpRowIDs is reused
							// Uses pooled buffer to avoid per-segment allocations
							var cloned imetadata.FilterResult
							cloned, rowBuf = fr.CloneInto(rowBuf)
							filters = append(filters, segFilter{seg: seg, fr: cloned, ts: ts})
						}
						// else: fr.IsEmpty() = no matches in this segment = skip (valid)
					}
				}

				// Use FilterResult strategy if below threshold OR user forced pre-filtering
				if possible && (forcePreFilter || totalHits <= adaptiveThreshold) {
					usedBitmapStrategy = true

					// Execute Brute Force with Early Cutoff using FilterResult
					var scoreFunc func([]float32, []float32) float32
					if e.metric == distance.MetricL2 {
						scoreFunc = distance.SquaredL2
					} else {
						scoreFunc = distance.Dot
					}

					// Batch size for processing - balance between I/O batching and early cutoff
					const batchSize = 64

					for _, sf := range filters {
						// Step 1: Get row IDs directly from FilterResult
						// For FilterRows mode, this is zero-copy
						// For FilterBitmap mode, uses ToArrayInto
						var allRowIDs []uint32
						if sf.fr.Mode() == imetadata.FilterRows {
							// Zero-copy: direct access to rows slice
							allRowIDs = sf.fr.Rows()
						} else {
							// Bitmap mode: extract via ToArrayInto
							allRowIDs = sf.fr.ToArrayInto(s.ScratchIDs[:0])
						}

						if len(allRowIDs) == 0 {
							continue
						}

						// Step 2: Batch filter tombstones using MatchesBatch (much faster than per-ID)
						if sf.ts != nil {
							if cap(s.ScratchBools) < len(allRowIDs) {
								s.ScratchBools = make([]bool, len(allRowIDs))
							}
							s.ScratchBools = s.ScratchBools[:len(allRowIDs)]
							sf.ts.MatchesBatch(allRowIDs, s.ScratchBools)

							// Compact: keep only alive IDs (may need to copy if FilterRows mode)
							if sf.fr.Mode() == imetadata.FilterRows {
								// FilterRows mode: need to copy alive IDs to scratch to avoid mutating original
								writeIdx := 0
								scratchCopy := s.ScratchIDs[:0]
								for i, alive := range s.ScratchBools {
									if alive {
										scratchCopy = append(scratchCopy, allRowIDs[i])
										writeIdx++
									}
								}
								allRowIDs = scratchCopy
								s.ScratchIDs = scratchCopy
							} else {
								// Bitmap mode: allRowIDs is from ScratchIDs, can mutate in place
								writeIdx := 0
								for i, alive := range s.ScratchBools {
									if alive {
										allRowIDs[writeIdx] = allRowIDs[i]
										writeIdx++
									}
								}
								allRowIDs = allRowIDs[:writeIdx]
								s.ScratchIDs = allRowIDs
							}
						}

						if len(allRowIDs) == 0 {
							continue
						}

						// Step 3: Sort rows for sequential mmap access (cache-friendly I/O)
						// FilterRows mode is already sorted (from roaring or sorted evaluation)
						// After tombstone compaction, order may be preserved but sort is O(n log n)
						// and mmap sequential access gains (~2-3x I/O throughput) outweigh sort cost
						if len(allRowIDs) > 64 && !slices.IsSorted(allRowIDs) {
							slices.Sort(allRowIDs)
						}

						// Step 4: Process in batches with early cutoff
						for start := 0; start < len(allRowIDs); start += batchSize {
							// Check context cancellation periodically
							if start > 0 && ctx.Err() != nil {
								yield(model.Candidate{}, ctx.Err())
								return
							}

							end := min(start+batchSize, len(allRowIDs))
							batchIDs := allRowIDs[start:end]

							// Ensure ScratchVecBuf is large enough for this batch
							vecBufSize := len(batchIDs) * e.dim
							if cap(s.ScratchVecBuf) < vecBufSize {
								s.ScratchVecBuf = make([]float32, vecBufSize)
							}
							s.ScratchVecBuf = s.ScratchVecBuf[:vecBufSize]

							// Fetch vectors using zero-alloc path
							validMask, err := sf.seg.FetchVectorsInto(ctx, batchIDs, e.dim, s.ScratchVecBuf)
							if err != nil {
								yield(model.Candidate{}, err)
								return
							}

							// Compute scores for batch
							for i := range batchIDs {
								// Check validity (nil mask means all valid)
								if len(validMask) > 0 && !validMask[i] {
									continue
								}

								vec := s.ScratchVecBuf[i*e.dim : (i+1)*e.dim]
								dist := scoreFunc(qExec, vec)
								s.FilterGateStats.DistanceComputations++
								s.FilterGateStats.CandidatesEvaluated++

								c := searcher.InternalCandidate{
									SegmentID: uint32(sf.seg.ID()),
									RowID:     batchIDs[i],
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
						}
					}
				}
			}

			if !usedBitmapStrategy {
				// Segment Search: Choose between sequential and parallel based on selectivity
				// At high selectivity (skipBitmapForHNSW = true), parallel search causes
				// goroutine coordination overhead to dominate (85% pthread_cond_wait).
				// Sequential search avoids this overhead entirely.
				numSegments := len(snap.sortedSegments)
				if snap.active != nil {
					numSegments++ // Include memtable
				}

				// FIX: Use sequential search when selectivity is high
				// This avoids goroutine fan-out and scheduler collapse
				useSequentialSearch := skipBitmapForHNSW || numSegments <= 2

				// Pass selectivity hint to HNSW for adaptive traversal
				// This enables unfiltered+post-filter mode at high selectivity
				if maxSelectivity > 0 {
					options.Selectivity = maxSelectivity
				}

				if useSequentialSearch {
					// SEQUENTIAL SEARCH: Process segments one by one
					// This avoids goroutine coordination overhead at high selectivity

					// Search MemTable first
					if snap.active != nil {
						if err := snap.active.Search(ctx, qExec, searchK, activeFilter, options, s); err != nil {
							yield(model.Candidate{}, err)
							return
						}
					}

					// Search immutable segments sequentially
					for i, seg := range snap.sortedSegments {
						// SEGMENT PRUNING: Skip segments that can be pruned based on manifest stats
						if options.Filter != nil && e.canPruneSegment(seg.ID(), options.Filter) {
							continue
						}

						var f segment.Filter
						if v := s.SegmentFilters[i]; v != nil {
							f = v.(segment.Filter)
						}

						if err := seg.Search(ctx, qExec, searchK, f, options, s); err != nil {
							yield(model.Candidate{}, err)
							return
						}
					}
				} else {
					// PARALLEL SEARCH: Fan out to goroutines for low selectivity
					// Only beneficial when:
					// 1. Selectivity is low (filter prunes most candidates)
					// 2. Multiple segments to search (parallelism helps)

					concurrency := runtime.GOMAXPROCS(0)
					if concurrency > numSegments {
						concurrency = numSegments
					}
					if concurrency == 0 {
						concurrency = 1
					}

					var wg sync.WaitGroup

					// OPTIMIZATION: Reuse semaphore channel when capacity is sufficient
					sem := s.SemChan
					if cap(sem) < concurrency {
						sem = make(chan struct{}, concurrency)
						s.SemChan = sem
					}

					// Storage for results to avoid locking the main heap
					// OPTIMIZATION: Use pooled result buffers to avoid allocation
					maxRes := numSegments * searchK
					if cap(s.ParallelResults) < maxRes {
						// Grow geometrically to reduce future reallocations
						newCap := max(maxRes, cap(s.ParallelResults)*2)
						s.ParallelResults = make([]searcher.InternalCandidate, newCap)
					}

					if cap(s.ParallelSlices) < numSegments {
						// Grow geometrically to reduce future reallocations
						newCap := max(numSegments, cap(s.ParallelSlices)*2)
						s.ParallelSlices = make([][]searcher.InternalCandidate, newCap)
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
					// With segment pruning: skip segments that can be pruned based on manifest stats
					for i, seg := range snap.sortedSegments {
						// SEGMENT PRUNING: Skip segments that can be pruned based on manifest stats
						// This is O(1) per segment and avoids HNSW traversal entirely
						if options.Filter != nil && e.canPruneSegment(seg.ID(), options.Filter) {
							// Segment can be skipped - no possible matches based on stats
							continue
						}

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
				} // end parallel search else block
			}
		}

	rerank:
		// 2. Extract candidates for reranking
		s.CandidateBuffer = s.CandidateBuffer[:0]
		for s.Heap.Len() > 0 {
			s.CandidateBuffer = append(s.CandidateBuffer, s.Heap.Pop())
		}

		// 3. Rerank - use package-level comparator to avoid closure allocation
		slices.SortFunc(s.CandidateBuffer, cmpInternalCandidateBySegment)

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
		// Pre-allocate cols with capacity 3 to avoid multiple slice growths
		cols := s.ScratchCols[:0]
		if options.IncludeVector {
			cols = append(cols, "vector")
		}
		if options.IncludeMetadata {
			cols = append(cols, "metadata")
		}
		if options.IncludePayload {
			cols = append(cols, "payload")
		}

		slices.SortFunc(s.Results, cmpCandidateBySegment)

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
		// Use package-level comparator to avoid closure allocation.
		if descending {
			slices.SortFunc(s.Results, cmpCandidateByScoreDesc)
		} else {
			slices.SortFunc(s.Results, cmpCandidateByScoreAsc)
		}

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

		// Populate internal stats from searcher before returning
		// Note: TotalDurationMicros, SegmentsSearched, Strategy are set by Search()
		if options.Stats != nil {
			options.Stats.DistanceComputations = int64(s.FilterGateStats.DistanceComputations)
			options.Stats.DistanceShortCircuits = int64(s.FilterGateStats.DistanceShortCircuits)
			options.Stats.NodesVisited = int64(s.FilterGateStats.NodesVisited)
			options.Stats.CandidatesEvaluated = int64(s.FilterGateStats.CandidatesEvaluated)
			options.Stats.BruteForceSegments = s.FilterGateStats.BruteForceSegments
			options.Stats.HNSWSegments = s.FilterGateStats.HNSWSegments
			options.Stats.FilterPassRate = s.FilterGateStats.FilterPassRate()
			options.Stats.FilterTimeMicros = s.FilterGateStats.FilterTimeNanos / 1000
			options.Stats.SearchTimeMicros = s.FilterGateStats.SearchTimeNanos / 1000
		}
	}
}

// Search performs a k-NN search and returns a slice of candidates.
// This is a convenience wrapper around SearchIter.
// If options.Stats is non-nil, it will be populated with query execution statistics.
func (e *Engine) Search(ctx context.Context, q []float32, k int, opts ...func(*model.SearchOptions)) ([]model.Candidate, error) {
	start := time.Now()

	// Parse options to check if stats collection is requested
	options := model.SearchOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	// Pre-allocate slice
	res := make([]model.Candidate, 0, k)

	next := e.SearchIter(ctx, q, k, opts...)
	for c, err := range next {
		if err != nil {
			return nil, err
		}
		res = append(res, c)
	}

	// Populate stats if requested
	if options.Stats != nil {
		options.Stats.TotalDurationMicros = time.Since(start).Microseconds()
		options.Stats.CandidatesReturned = len(res)
		// SegmentsSearched is derived from actual search activity (already set by SearchIter)
		options.Stats.SegmentsSearched = options.Stats.BruteForceSegments + options.Stats.HNSWSegments
		if options.Filter == nil {
			options.Stats.Strategy = "hnsw"
		} else {
			options.Stats.Strategy = "hnsw_filtered"
		}
	}

	return res, nil
}

// segmentWithFilterResult is an interface for segments that support zero-alloc filter evaluation.
type segmentWithFilterResult interface {
	EvaluateFilterResult(ctx context.Context, filter *metadata.FilterSet, qs *imetadata.QueryScratch) (imetadata.FilterResult, error)
}

// evaluateSegmentFilterResult tries the zero-alloc EvaluateFilterResult path.
// Falls back to the bitmap path if the segment doesn't support it.
func (e *Engine) evaluateSegmentFilterResult(ctx context.Context, seg segment.Segment, filter *metadata.FilterSet, qs *imetadata.QueryScratch) (imetadata.FilterResult, error) {
	// Try zero-alloc path first
	if sfr, ok := seg.(segmentWithFilterResult); ok {
		return sfr.EvaluateFilterResult(ctx, filter, qs)
	}

	// Fallback: use legacy bitmap path and convert to FilterResult
	bm, err := seg.EvaluateFilter(ctx, filter)
	if err != nil {
		return imetadata.EmptyResult(), err
	}
	if bm == nil {
		return imetadata.AllResult(), nil // matches all - no filtering
	}
	if bm.Cardinality() == 0 {
		// Explicit empty bitmap = no matches
		if lb, ok := bm.(*imetadata.LocalBitmap); ok {
			imetadata.PutPooledBitmap(lb)
		}
		return imetadata.EmptyResult(), nil
	}

	// Convert bitmap to FilterResult
	// For small results, extract to rows mode (more cache-friendly)
	// For large results, use QueryBitmap for SIMD execution
	const rowsThreshold = 1024
	card := bm.Cardinality()

	if card <= rowsThreshold {
		out := qs.TmpRowIDs[:0]
		out = bm.ToArrayInto(out)
		qs.TmpRowIDs = out

		// Return the bitmap to pool if it's a LocalBitmap
		if lb, ok := bm.(*imetadata.LocalBitmap); ok {
			imetadata.PutPooledBitmap(lb)
		}

		return imetadata.RowsResult(out), nil
	}

	// Large result: convert to QueryBitmap for SIMD execution
	// Extract to temporary slice first, then bulk-add to QueryBitmap
	// This is faster than ForEach + Add because AddMany batches block tracking
	out := qs.TmpRowIDs[:0]
	out = bm.ToArrayInto(out)
	qs.TmpRowIDs = out

	// Return the source bitmap to pool BEFORE populating QueryBitmap
	if lb, ok := bm.(*imetadata.LocalBitmap); ok {
		imetadata.PutPooledBitmap(lb)
	}

	// Bulk populate QueryBitmap from extracted rows
	qs.Tmp2.Clear()
	qs.Tmp2.AddMany(out)

	return imetadata.QueryBitmapResult(qs.Tmp2), nil
}
