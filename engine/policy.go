package engine

import "github.com/hupe1980/vecgo/model"

// CompactionPolicy determines which segments should be compacted.
type CompactionPolicy interface {
	// Pick selects segments to compact.
	// Returns a list of segment IDs. If the list is empty, no compaction is needed.
	Pick(segments []model.SegmentID) []model.SegmentID
}

// TieredCompactionPolicy implements a simple size-tiered compaction strategy.
// It triggers compaction when there are at least `Threshold` segments.
type TieredCompactionPolicy struct {
	Threshold int
}

func (p *TieredCompactionPolicy) Pick(segments []model.SegmentID) []model.SegmentID {
	if len(segments) >= p.Threshold {
		// For simplicity, just pick all of them for now.
		// A real implementation would group them by size/level.
		return segments
	}
	return nil
}
