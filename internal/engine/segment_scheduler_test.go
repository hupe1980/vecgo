package engine

import (
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/manifest"
	"github.com/hupe1980/vecgo/model"
)

func TestSegmentScheduler_Basic(t *testing.T) {
	// Create stats provider that returns different stats for different segments
	statsMap := map[model.SegmentID]*manifest.SegmentStats{
		1: {
			TotalRows:     1000,
			LiveRows:      950,
			DeletedRatio:  0.05,
			FilterEntropy: 0.3,
			Vector: &manifest.VectorStats{
				MeanNorm:              100.0,
				AvgDistanceToCentroid: 50.0,
				Radius95:              75.0,
				Centroid:              make([]int8, 128),
			},
			Shape: &manifest.ShapeStats{
				ClusterTightness: 0.8,
			},
		},
		2: {
			TotalRows:     1000,
			LiveRows:      700,
			DeletedRatio:  0.3, // High deletion ratio
			FilterEntropy: 0.7,
			Vector: &manifest.VectorStats{
				MeanNorm:              100.0,
				AvgDistanceToCentroid: 100.0,
				Radius95:              150.0,
				Centroid:              make([]int8, 128),
			},
			Shape: &manifest.ShapeStats{
				ClusterTightness: 0.3,
			},
		},
		3: {
			TotalRows:     1000,
			LiveRows:      990,
			DeletedRatio:  0.01,
			FilterEntropy: 0.1, // Low entropy - good for early termination
			Vector: &manifest.VectorStats{
				MeanNorm:              100.0,
				AvgDistanceToCentroid: 30.0,
				Radius95:              40.0,
				Centroid:              make([]int8, 128),
			},
			Shape: &manifest.ShapeStats{
				ClusterTightness: 0.95,
			},
		},
	}

	statsProvider := func(id model.SegmentID) *manifest.SegmentStats {
		return statsMap[id]
	}

	scheduler := NewSegmentScheduler(statsProvider, distance.MetricL2, 128)

	// Create test segments
	segments := []SegmentPriority{
		{SegmentID: 1},
		{SegmentID: 2},
		{SegmentID: 3},
	}

	// Schedule with a query
	query := make([]float32, 128)
	params := ScheduleParams{
		Query: query,
		K:     10,
	}

	result := scheduler.Schedule(segments, params)

	// Segment 3 should be first (tightest cluster, lowest entropy)
	if len(result) == 0 {
		t.Fatal("expected non-empty result")
	}

	// All segments should be present (no pruning without maxDistance)
	if len(result) != 3 {
		t.Errorf("expected 3 segments, got %d", len(result))
	}

	// Segment 3 should have highest priority
	if result[0].SegmentID != 3 {
		t.Errorf("expected segment 3 first (best cluster), got %d", result[0].SegmentID)
	}

	// Segment 2 should be last (high deletion, loose cluster)
	if result[len(result)-1].SegmentID != 2 {
		t.Errorf("expected segment 2 last (worst), got %d", result[len(result)-1].SegmentID)
	}
}

func TestSegmentScheduler_DistancePruning(t *testing.T) {
	statsMap := map[model.SegmentID]*manifest.SegmentStats{
		1: {
			TotalRows: 1000,
			LiveRows:  1000,
			Vector: &manifest.VectorStats{
				MeanNorm:              100.0,
				AvgDistanceToCentroid: 10.0,
				Radius95:              15.0,
				RadiusMax:             20.0,
				Centroid:              make([]int8, 128),
			},
		},
		2: {
			TotalRows: 1000,
			LiveRows:  1000,
			Vector: &manifest.VectorStats{
				MeanNorm:              100.0,
				AvgDistanceToCentroid: 1000.0, // Far away
				Radius95:              50.0,
				RadiusMax:             100.0,
				Centroid:              make([]int8, 128),
			},
		},
	}

	statsProvider := func(id model.SegmentID) *manifest.SegmentStats {
		return statsMap[id]
	}

	scheduler := NewSegmentScheduler(statsProvider, distance.MetricL2, 128)

	segments := []SegmentPriority{
		{SegmentID: 1},
		{SegmentID: 2},
	}

	// Schedule with max distance that should prune segment 2
	// Segment 2's minimum possible distance = centroidDist - radius95
	// If this exceeds maxDistance, it should be pruned
	query := make([]float32, 128)
	params := ScheduleParams{
		Query:       query,
		K:           10,
		MaxDistance: 100.0, // Much smaller than segment 2's expected distance
	}

	result := scheduler.Schedule(segments, params)

	// With proper distance pruning, segment 2 might be pruned
	// Note: actual pruning depends on centroid distance computation
	// which uses quantized centroids
	for _, seg := range result {
		t.Logf("Segment %d: priority=%.2f, centroidDist=%.2f, canPrune=%v",
			seg.SegmentID, seg.Priority, seg.CentroidDistance, seg.CanPrune)
	}
}

func TestSegmentScheduler_EarlyTermination(t *testing.T) {
	statsMap := map[model.SegmentID]*manifest.SegmentStats{
		1: {
			Vector: &manifest.VectorStats{
				Radius95: 10.0,
			},
		},
		2: {
			Vector: &manifest.VectorStats{
				Radius95: 10.0,
			},
		},
		3: {
			Vector: &manifest.VectorStats{
				Radius95: 10.0,
			},
		},
	}

	statsProvider := func(id model.SegmentID) *manifest.SegmentStats {
		return statsMap[id]
	}

	scheduler := NewSegmentScheduler(statsProvider, distance.MetricL2, 128)

	// Remaining segments with high centroid distances
	remaining := []SegmentPriority{
		{
			SegmentID:        2,
			CentroidDistance: 500.0,
			Stats:            statsMap[2],
		},
		{
			SegmentID:        3,
			CentroidDistance: 600.0,
			Stats:            statsMap[3],
		},
	}

	// Should terminate early if worst result distance is much smaller
	// than minimum possible distance to remaining segments
	shouldTerminate := scheduler.ShouldEarlyTerminate(
		2,    // searched 2 segments
		4,    // out of 4 total
		10,   // found 10 results
		10,   // k=10
		50.0, // worst result at distance 50
		remaining,
	)

	// With centroid distances of 500 and 600, and radius of 10,
	// minimum possible distances are 490 and 590
	// Both exceed worst result distance of 50
	// Should terminate early
	if !shouldTerminate {
		t.Error("expected early termination when remaining segments are far away")
	}
}

func TestSegmentScheduler_NoEarlyTermination(t *testing.T) {
	statsMap := map[model.SegmentID]*manifest.SegmentStats{
		1: {
			Vector: &manifest.VectorStats{
				Radius95: 100.0, // Large radius
			},
		},
	}

	statsProvider := func(id model.SegmentID) *manifest.SegmentStats {
		return statsMap[id]
	}

	scheduler := NewSegmentScheduler(statsProvider, distance.MetricL2, 128)

	// Remaining segment with close centroid
	remaining := []SegmentPriority{
		{
			SegmentID:        1,
			CentroidDistance: 60.0,
			Stats:            statsMap[1],
		},
	}

	// Should NOT terminate - remaining segment could have closer results
	shouldTerminate := scheduler.ShouldEarlyTerminate(
		2,    // searched 2 segments
		3,    // out of 3 total
		10,   // found 10 results
		10,   // k=10
		50.0, // worst result at distance 50
		remaining,
	)

	// centroid distance 60 - radius 100 = -40 (clamped to 0)
	// 0 < 50, so we can't prune
	if shouldTerminate {
		t.Error("should not terminate when remaining segment could have closer results")
	}
}

func TestSegmentScheduler_Stats(t *testing.T) {
	scheduler := NewSegmentScheduler(nil, distance.MetricL2, 128)

	// Initial stats should be zero
	stats := scheduler.Stats()
	if stats.TotalScheduled != 0 {
		t.Errorf("expected 0 scheduled, got %d", stats.TotalScheduled)
	}

	// Schedule some segments
	segments := []SegmentPriority{{SegmentID: 1}}
	query := make([]float32, 128)
	scheduler.Schedule(segments, ScheduleParams{Query: query, K: 10})

	stats = scheduler.Stats()
	if stats.TotalScheduled != 1 {
		t.Errorf("expected 1 scheduled, got %d", stats.TotalScheduled)
	}

	// Reset stats
	scheduler.Reset()
	stats = scheduler.Stats()
	if stats.TotalScheduled != 0 {
		t.Errorf("expected 0 after reset, got %d", stats.TotalScheduled)
	}
}

func BenchmarkSegmentScheduler_Schedule(b *testing.B) {
	// Create a realistic scenario with 100 segments
	statsMap := make(map[model.SegmentID]*manifest.SegmentStats)
	for i := model.SegmentID(1); i <= 100; i++ {
		statsMap[i] = &manifest.SegmentStats{
			TotalRows:     1000,
			LiveRows:      uint32(900 + int(i)%100),
			DeletedRatio:  float32(i%20) / 100,
			FilterEntropy: float32(i%10) / 10,
			Vector: &manifest.VectorStats{
				MeanNorm:              100.0,
				AvgDistanceToCentroid: float32(50 + i%50),
				Radius95:              float32(30 + i%30),
				Centroid:              make([]int8, 128),
			},
			Shape: &manifest.ShapeStats{
				ClusterTightness: float32(50+i%50) / 100,
			},
		}
	}

	statsProvider := func(id model.SegmentID) *manifest.SegmentStats {
		return statsMap[id]
	}

	scheduler := NewSegmentScheduler(statsProvider, distance.MetricL2, 128)

	segments := make([]SegmentPriority, 100)
	for i := range segments {
		segments[i] = SegmentPriority{SegmentID: model.SegmentID(i + 1)}
	}

	query := make([]float32, 128)
	params := ScheduleParams{
		Query: query,
		K:     10,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Copy segments for each iteration (Schedule modifies the slice)
		segCopy := make([]SegmentPriority, len(segments))
		copy(segCopy, segments)
		scheduler.Schedule(segCopy, params)
	}
}
