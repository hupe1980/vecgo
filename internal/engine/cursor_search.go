package engine

import (
	"context"
	"time"

	"github.com/hupe1980/vecgo/distance"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
)

// CursorSearchConfig holds configuration for cursor-based search.
type CursorSearchConfig struct {
	// SelectivityCutoff: above this threshold, use HNSW instead of brute-force.
	// Default: 0.30 (30%)
	SelectivityCutoff float64

	// BatchSize: number of vectors to fetch per I/O batch.
	// Default: 64
	BatchSize int

	// ForcePreFilter: if true, always use pre-filtering even at high selectivity.
	ForcePreFilter bool
}

// DefaultCursorSearchConfig returns sensible defaults.
func DefaultCursorSearchConfig() CursorSearchConfig {
	return CursorSearchConfig{
		SelectivityCutoff: 0.30,
		BatchSize:         64,
	}
}

// CursorFilterable is the interface for segments that support cursor-based filtering.
type CursorFilterable interface {
	FilterCursor(filter *metadata.FilterSet) imetadata.FilterCursor
}

// GraphIndexSegment is an optional interface for segments that have graph-based indexes.
// Used to distinguish between flat segments (brute-force) and graph segments (HNSW/DiskANN).
type GraphIndexSegment interface {
	HasGraphIndex() bool
}

// SegmentUnwrapper is an interface for wrappers (like RefCountedSegment) that embed a segment.
type SegmentUnwrapper interface {
	UnwrapSegment() segment.Segment
}

// segmentHasGraphIndex checks if a segment has a graph-based index (HNSW, DiskANN).
// Flat segments return false - they should always use cursor for filtered search.
// This function handles wrapper types (RefCountedSegment) by unwrapping them.
func segmentHasGraphIndex(seg segment.Segment) bool {
	// Unwrap if this is a wrapper (RefCountedSegment embeds segment.Segment)
	if unwrapper, ok := seg.(SegmentUnwrapper); ok {
		seg = unwrapper.UnwrapSegment()
	}

	if gi, ok := seg.(GraphIndexSegment); ok {
		return gi.HasGraphIndex()
	}
	// Default: assume segment has graph (conservative - falls back to HNSW)
	// Memtable always has HNSW, DiskANN segments have Vamana graph
	return true
}

// searchSegmentWithCursor performs a cursor-based filtered search on a segment.
// This is the zero-allocation hot path that:
//   - Uses push-based iteration (no bitmap materialization)
//   - Batches vector fetches (amortizes I/O)
//   - Supports early termination
//   - Applies tombstone filtering inline
//
// For FLAT segments: ALWAYS uses cursor (avoids expensive per-row MatchesBinary).
// For HNSW segments: Uses selectivity cutoff to decide (graph traversal vs cursor).
//
// Returns true if the cursor was used, false if HNSW should be used instead.
func (e *Engine) searchSegmentWithCursor(
	ctx context.Context,
	seg segment.Segment,
	cursor imetadata.FilterCursor,
	tombstone segment.Filter,
	config CursorSearchConfig,
	s *searcher.Searcher,
	query []float32,
	searchK int,
) (bool, error) {
	// Check selectivity
	if cursor.IsAll() {
		return false, nil // Use HNSW
	}

	if cursor.IsEmpty() {
		return true, nil // No matches, done
	}

	// Estimate selectivity
	segSize := seg.RowCount()
	selectivity := float64(cursor.EstimateCardinality()) / float64(segSize)

	// High selectivity routing decision:
	// - For FLAT segments: ALWAYS use cursor (avoid expensive MatchesBinary per-row)
	// - For HNSW segments: Use HNSW with post-filter when selectivity > cutoff
	//
	// Why: Flat segment brute-force with per-row filter (MatchesBinary) is 53%+ CPU
	// in Zipfian workloads. Cursor path gets matching row IDs first, then computes
	// distances only for matches - always faster than N*MatchesBinary calls.
	if !config.ForcePreFilter && selectivity > config.SelectivityCutoff {
		// Check if this is a flat segment (no graph traversal benefit)
		// Flat segments should ALWAYS use cursor to avoid per-row MatchesBinary
		if !segmentHasGraphIndex(seg) {
			// Flat segment: proceed with cursor (it's faster than brute-force + filter)
			// The cursor will iterate all matches, which is fine for high selectivity
		} else {
			// HNSW/DiskANN segment: use graph traversal with post-filter
			return false, nil
		}
	}

	// Track search timing
	searchStart := time.Now()
	defer func() {
		s.FilterGateStats.SearchTimeNanos += time.Since(searchStart).Nanoseconds()
	}()

	// Set up state
	batchSize := config.BatchSize
	if batchSize <= 0 {
		batchSize = 64
	}

	// Track that this is a brute-force segment search
	s.FilterGateStats.BruteForceSegments++

	// Ensure batch buffer capacity
	if cap(s.ScratchIDs) < batchSize {
		s.ScratchIDs = make([]uint32, 0, batchSize)
	}
	batchIDs := s.ScratchIDs[:0]

	// Score function
	var scoreFunc func([]float32, []float32) float32
	if e.metric == distance.MetricL2 {
		scoreFunc = distance.SquaredL2
	} else {
		scoreFunc = distance.Dot
	}

	segID := uint32(seg.ID())

	// Try zero-copy direct access first (for immutable segments)
	// This avoids memmove entirely by computing distance directly on mmap'd memory
	testVec := seg.FetchVectorDirect(0)
	supportsDirectAccess := testVec != nil

	if supportsDirectAccess {
		// Zero-copy path: compute distance directly on mmap'd vectors
		var processErr error
		cursor.ForEach(func(rowID uint32) bool {
			// Apply tombstone filter
			if tombstone != nil && !tombstone.Matches(rowID) {
				return true // continue, skip tombstoned
			}

			vec := seg.FetchVectorDirect(rowID)
			if vec == nil {
				return true // skip invalid
			}

			dist := scoreFunc(query, vec)
			s.FilterGateStats.DistanceComputations++
			s.FilterGateStats.CandidatesEvaluated++

			c := searcher.InternalCandidate{
				SegmentID: segID,
				RowID:     rowID,
				Score:     dist,
				Approx:    false,
			}

			s.Heap.TryPushBounded(c, searchK)

			return true
		})

		return true, processErr
	}

	// Fallback: batch fetch with copy (for memtable and segments without direct access)
	// Process function for batches
	processBatch := func() error {
		if len(batchIDs) == 0 {
			return nil
		}

		// Ensure vector buffer capacity
		vecBufSize := len(batchIDs) * e.dim
		if cap(s.ScratchVecBuf) < vecBufSize {
			s.ScratchVecBuf = make([]float32, vecBufSize)
		}
		s.ScratchVecBuf = s.ScratchVecBuf[:vecBufSize]

		// Fetch vectors
		validMask, err := seg.FetchVectorsInto(ctx, batchIDs, e.dim, s.ScratchVecBuf)
		if err != nil {
			return err
		}

		// Score and push to heap
		for i, rowID := range batchIDs {
			if len(validMask) > 0 && !validMask[i] {
				continue
			}

			vec := s.ScratchVecBuf[i*e.dim : (i+1)*e.dim]
			dist := scoreFunc(query, vec)
			s.FilterGateStats.DistanceComputations++
			s.FilterGateStats.CandidatesEvaluated++

			c := searcher.InternalCandidate{
				SegmentID: segID,
				RowID:     rowID,
				Score:     dist,
				Approx:    false,
			}

			s.Heap.TryPushBounded(c, searchK)
		}

		return nil
	}

	// Iterate with cursor
	var processErr error
	cursor.ForEach(func(rowID uint32) bool {
		// Apply tombstone filter
		if tombstone != nil && !tombstone.Matches(rowID) {
			return true // continue, skip tombstoned
		}

		batchIDs = append(batchIDs, rowID)

		// Process batch when full
		if len(batchIDs) >= batchSize {
			if err := processBatch(); err != nil {
				processErr = err
				return false
			}
			batchIDs = batchIDs[:0]
		}

		return true
	})

	if processErr != nil {
		return true, processErr
	}

	// Process remaining batch
	if len(batchIDs) > 0 {
		if err := processBatch(); err != nil {
			return true, err
		}
	}

	// Store back for capacity tracking
	s.ScratchIDs = batchIDs[:0]

	return true, nil
}

// SegmentWithFilterCursor is the interface for segments that support cursor-based filtering.
// This is the preferred interface for zero-allocation filter evaluation.
type SegmentWithFilterCursor interface {
	segment.Segment
	FilterCursor(filter *metadata.FilterSet) imetadata.FilterCursor
}

// MemTableFilterCursor is the interface for memtables that support cursor filtering.
type MemTableFilterCursor interface {
	segment.Segment
	FilterCursor(filter *metadata.FilterSet) imetadata.FilterCursor
}
