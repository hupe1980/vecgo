package engine

import (
	"slices"

	"github.com/hupe1980/vecgo/model"
)

// SegmentStats holds metadata about a segment needed for compaction decisions.
type SegmentStats struct {
	ID    model.SegmentID
	Size  int64
	Level int
	MinID model.ID
	MaxID model.ID
}

// CompactionTask describes a compaction unit of work.
type CompactionTask struct {
	Segments    []model.SegmentID
	TargetLevel int
}

// CompactionPolicy determines which segments should be compacted.
type CompactionPolicy interface {
	// Pick selects segments to compact.
	// Returns a task or nil if no compaction is needed.
	Pick(segments []SegmentStats) *CompactionTask
}

// TieredCompactionPolicy implements a simple size-tiered compaction strategy.
// It triggers compaction when there are at least `Threshold` segments.
type TieredCompactionPolicy struct {
	Threshold int
}

func (p *TieredCompactionPolicy) Pick(segments []SegmentStats) *CompactionTask {
	if len(segments) >= p.Threshold {
		// For simplicity, just pick all of them for now.
		ids := make([]model.SegmentID, len(segments))
		for i, s := range segments {
			ids[i] = s.ID
		}
		return &CompactionTask{
			Segments:    ids,
			TargetLevel: 1, // Promote to Level 1 to indicate "compacted"
		}
	}
	return nil
}

// BoundedSizeTieredPolicy implements a size-tiered compaction strategy with explicit bounds.
// - Segment size buckets: [0-10MB], [10-100MB], [100MB-1GB], [1GB+]
// - Compact within bucket only when threshold exceeded
// - Never compact segments spanning multiple buckets in single operation
// - Max compaction bytes: 2GB hard limit
type BoundedSizeTieredPolicy struct {
	Threshold int
}

func (p *BoundedSizeTieredPolicy) Pick(segments []SegmentStats) *CompactionTask {
	// 1. Group segments by size bucket
	buckets := make(map[int][]SegmentStats)
	for _, s := range segments {
		bucket := p.getBucket(s.Size)
		buckets[bucket] = append(buckets[bucket], s)
	}

	// 2. Check each bucket for compaction candidates
	// Iterate buckets from smallest to largest
	for i := 0; i < 4; i++ {
		bucketSegs := buckets[i]
		if len(bucketSegs) >= p.Threshold {
			// Sort by ID (age proxy)
			slices.SortFunc(bucketSegs, func(a, b SegmentStats) int {
				if a.ID < b.ID {
					return -1
				}
				if a.ID > b.ID {
					return 1
				}
				return 0
			})

			// Pick first N segments that fit within 2GB limit
			var toCompact []model.SegmentID
			var totalSize int64
			const maxCompactionSize = 2 * 1024 * 1024 * 1024 // 2GB

			for _, s := range bucketSegs {
				if totalSize+s.Size > maxCompactionSize {
					break
				}
				toCompact = append(toCompact, s.ID)
				totalSize += s.Size
			}

			if len(toCompact) >= 2 { // Need at least 2 to compact
				// Target level is the level of the bucket?
				// Or simply 0 if we don't strictly use levels yet?
				// To minimize changes, let's say target level depends on result size.
				// But simpler: just preserve current logic.
				return &CompactionTask{
					Segments:    toCompact,
					TargetLevel: i + 1, // Promote to next logical level (simplified)
				}
			}
		}
	}

	return nil
}

// LeveledCompactionPolicy implements a Level-Based compaction strategy.
// Features:
// - L0: Overlapping segments (flushed from MemTable). Size unconstrained.
// - L1..N: Non-overlapping segments (target size).
// - Strategy:
//   - If L0 count > 4 -> Compact all L0 -> L1.
//   - If size(L_i) > 10^i * BaseSize -> Select 1 segment from L_i + overlapping from L_{i+1} -> L_{i+1}.
//
// For VectorDB simplicity, we simplify "overlapping" to just "merge N segments from L_i to L_{i+1}".
type LeveledCompactionPolicy struct {
	L0Threshold int   // Number of files in L0 to trigger compaction (default 4)
	LevelRatio  int   // Growth ratio between levels (default 10)
	BaseSize    int64 // Target size of L1 (default 100MB)
	MaxLevels   int   // Maximum number of levels (default 7)
}

func NewLeveledCompactionPolicy() *LeveledCompactionPolicy {
	return &LeveledCompactionPolicy{
		L0Threshold: 4,
		LevelRatio:  10,
		BaseSize:    100 * 1024 * 1024, // 100MB
		MaxLevels:   7,
	}
}

func (p *LeveledCompactionPolicy) Pick(segments []SegmentStats) *CompactionTask {
	// Group by Level
	levels := make([][]SegmentStats, p.MaxLevels)
	for _, s := range segments {
		if s.Level < p.MaxLevels {
			levels[s.Level] = append(levels[s.Level], s)
		} else {
			// Treat max level overflow as max level
			levels[p.MaxLevels-1] = append(levels[p.MaxLevels-1], s)
		}
	}

	// 1. Check L0
	// If L0 has too many files, compact them all into L1.
	if len(levels[0]) >= p.L0Threshold {
		ids := make([]model.SegmentID, 0, len(levels[0]))
		// Sort L0 by Age (ID)
		slices.SortFunc(levels[0], func(a, b SegmentStats) int {
			return int(a.ID - b.ID)
		})

		for _, s := range levels[0] {
			ids = append(ids, s.ID)
		}
		return &CompactionTask{
			Segments:    ids,
			TargetLevel: 1,
		}
	}

	// 2. Check L1..N
	// For each level, calculate target size.
	// Target(L1) = BaseSize
	// Target(L_i) = Target(L_{i-1}) * Ratio
	targetSize := p.BaseSize

	for lvl := 1; lvl < p.MaxLevels-1; lvl++ {
		var currentSize int64
		for _, s := range levels[lvl] {
			currentSize += s.Size
		}

		if currentSize > targetSize {
			// Trigger compaction for this level.
			// Strategy: Pick the oldest segment in this level (lowest ID)
			// and merge it into the next level.
			// Note: Real classic LCS picks based on overlap with next level.
			// Here we pick oldest to maintain roughly time-order.

			slices.SortFunc(levels[lvl], func(a, b SegmentStats) int {
				return int(a.ID - b.ID)
			})

			// Pick simple strategy:
			// Take the oldest segment from L_current.
			// PLUS any overlapping segments from L_next?
			// Without ID range info, we can't determine overlap.
			// But we know we want to move data to L_next.
			// If we just move ONE segment, we increase L_next count (fragmentation).
			// We should merge it with something?
			// Since we lack range info in SegmentStats, we fall back to a "Tiering-like" move:
			// Pick N segments from L_current to form a decent size chunk for L_next?
			// OR: Just compact the Level if it's too big?

			// Let's implement full level compaction:
			// If Level is too big, take ALL segments in that level and push to next?
			// No, that's Tiering.

			// Approximation of LCS for unsorted segments:
			// Pick oldest segment.
			victim := levels[lvl][0]

			return &CompactionTask{
				Segments:    []model.SegmentID{victim.ID},
				TargetLevel: lvl + 1,
			}
		}

		targetSize *= int64(p.LevelRatio)
	}

	return nil
}

func (p *BoundedSizeTieredPolicy) getBucket(size int64) int {
	const (
		MB = 1024 * 1024
		GB = 1024 * MB
	)
	if size < 10*MB {
		return 0
	}
	if size < 100*MB {
		return 1
	}
	if size < 1*GB {
		return 2
	}
	return 3
}
