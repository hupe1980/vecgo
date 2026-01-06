package engine

import (
	"slices"

	"github.com/hupe1980/vecgo/model"
)

// SegmentStats holds metadata about a segment needed for compaction decisions.
type SegmentStats struct {
	ID   model.SegmentID
	Size int64
}

// CompactionPolicy determines which segments should be compacted.
type CompactionPolicy interface {
	// Pick selects segments to compact.
	// Returns a list of segment IDs. If the list is empty, no compaction is needed.
	Pick(segments []SegmentStats) []model.SegmentID
}

// TieredCompactionPolicy implements a simple size-tiered compaction strategy.
// It triggers compaction when there are at least `Threshold` segments.
type TieredCompactionPolicy struct {
	Threshold int
}

func (p *TieredCompactionPolicy) Pick(segments []SegmentStats) []model.SegmentID {
	if len(segments) >= p.Threshold {
		// For simplicity, just pick all of them for now.
		ids := make([]model.SegmentID, len(segments))
		for i, s := range segments {
			ids[i] = s.ID
		}
		return ids
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

func (p *BoundedSizeTieredPolicy) Pick(segments []SegmentStats) []model.SegmentID {
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
				return toCompact
			}
		}
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
