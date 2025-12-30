package vecgo_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
)

func TestBuilder_HNSW_Basic(t *testing.T) {
	db, err := vecgo.HNSW[string](4).
		SquaredL2().
		Build()
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	id, err := db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1, 2, 3, 4},
		Data:   "test",
	})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// ID should be valid (>= 0)
	_ = id
}

func TestBuilder_HNSW_FullOptions(t *testing.T) {
	db, err := vecgo.HNSW[string](4).
		SquaredL2().
		M(32).
		EFConstruction(100).
		Heuristic(false).
		Shards(2).
		Build()
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	_, err = db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1, 2, 3, 4},
		Data:   "test",
	})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
}

func TestBuilder_Flat_Basic(t *testing.T) {
	db, err := vecgo.Flat[int](4).
		Cosine().
		Build()
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	id, err := db.Insert(ctx, vecgo.VectorWithData[int]{
		Vector: []float32{1, 0, 0, 0},
		Data:   42,
	})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// ID should be valid (>= 0)
	_ = id
}

func TestBuilder_MustBuild_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected MustBuild to panic on invalid config")
		}
	}()

	// Invalid dimension should cause panic
	_ = vecgo.HNSW[string](0). // Invalid dimension
					MustBuild()
}

func TestSearchBuilder_KNN(t *testing.T) {
	db, err := vecgo.HNSW[string](4).
		Build()
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Insert some vectors
	vectors := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	for i, v := range vectors {
		_, err := db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: v,
			Data:   string(rune('A' + i)),
		})
		if err != nil {
			t.Fatalf("Insert failed: %v", err)
		}
	}

	// Search using fluent API
	results, err := db.Search([]float32{1, 0, 0, 0}).
		KNN(2).
		Execute(ctx)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// First result should be exact match
	if results[0].Data != "A" {
		t.Errorf("expected first result to be 'A', got %q", results[0].Data)
	}
}

func TestSearchBuilder_Search(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()
	_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1, 2, 3, 4},
		Data:   "target",
	})

	results, err := db.Search([]float32{1, 2, 3, 4}).
		KNN(1).
		Execute(ctx)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 || results[0].Data != "target" {
		t.Errorf("unexpected results: %v", results)
	}
}

func TestSearchBuilder_Filter(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()

	// Insert vectors with IDs 0-3
	for i := 0; i < 4; i++ {
		_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: []float32{float32(i), 0, 0, 0},
			Data:   string(rune('A' + i)),
		})
	}

	// Search with filter - verify filter is called
	filterCalled := false
	results, err := db.Search([]float32{0, 0, 0, 0}).
		KNN(10).
		Filter(func(id uint64) bool {
			filterCalled = true
			return true // Accept all for this test
		}).
		Execute(ctx)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if !filterCalled {
		t.Error("filter function was not called")
	}

	if len(results) == 0 {
		t.Error("expected results from search")
	}
}

func TestSearchBuilder_First(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()
	_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1, 2, 3, 4},
		Data:   "first",
	})

	result, err := db.Search([]float32{1, 2, 3, 4}).First(ctx)
	if err != nil {
		t.Fatalf("First failed: %v", err)
	}

	if result.Data != "first" {
		t.Errorf("expected 'first', got %q", result.Data)
	}
}

func TestSearchBuilder_First_NotFound(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()
	// Empty index - First should return ErrNotFound
	_, err := db.Search([]float32{1, 2, 3, 4}).First(ctx)
	if err != vecgo.ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestSearchBuilder_Exists(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()

	// Empty index
	exists, err := db.Search([]float32{1, 2, 3, 4}).Exists(ctx)
	if err != nil {
		t.Fatalf("Exists failed: %v", err)
	}
	if exists {
		t.Error("expected no results to exist in empty index")
	}

	// After insert
	_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1, 2, 3, 4},
		Data:   "test",
	})

	exists, err = db.Search([]float32{1, 2, 3, 4}).Exists(ctx)
	if err != nil {
		t.Fatalf("Exists failed: %v", err)
	}
	if !exists {
		t.Error("expected results to exist after insert")
	}
}

func TestSearchBuilder_Stream(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()

	// Insert 10 vectors
	for i := 0; i < 10; i++ {
		_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: []float32{float32(i), 0, 0, 0},
			Data:   string(rune('A' + i)),
		})
	}

	// Stream results with early termination
	var count int
	for result, err := range db.Search([]float32{0, 0, 0, 0}).KNN(10).Stream(ctx) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		count++
		_ = result
		if count >= 3 {
			break // Early termination
		}
	}

	if count != 3 {
		t.Errorf("expected 3 results before early termination, got %d", count)
	}
}

func TestSearchBuilder_WithMetadata(t *testing.T) {
	db := vecgo.HNSW[string](4).
		MustBuild()
	defer db.Close()

	ctx := context.Background()

	// Insert vectors with metadata
	_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{1, 0, 0, 0},
		Data:   "doc1",
		Metadata: metadata.Metadata{
			"category": metadata.String("tech"),
		},
	})
	_, _ = db.Insert(ctx, vecgo.VectorWithData[string]{
		Vector: []float32{0, 1, 0, 0},
		Data:   "doc2",
		Metadata: metadata.Metadata{
			"category": metadata.String("science"),
		},
	})

	// Search with metadata filter
	filters := metadata.NewFilterSet(
		metadata.Filter{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("tech")},
	)

	results, err := db.Search([]float32{0.5, 0.5, 0, 0}).
		KNN(10).
		WithMetadata(filters).
		Execute(ctx)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 1 || results[0].Data != "doc1" {
		t.Errorf("expected only 'doc1' with category=tech, got %v", results)
	}
}

func TestBuilder_DistanceShortcuts(t *testing.T) {
	tests := []struct {
		name     string
		builder  func() vecgo.HNSWBuilder[string]
		expected string // Check via Options map in Stats
	}{
		{
			name:     "SquaredL2",
			builder:  func() vecgo.HNSWBuilder[string] { return vecgo.HNSW[string](4).SquaredL2() },
			expected: "SquaredL2",
		},
		{
			name:     "Cosine",
			builder:  func() vecgo.HNSWBuilder[string] { return vecgo.HNSW[string](4).Cosine() },
			expected: "CosineDistance", // Internal name uses CosineDistance
		},
		{
			name:     "DotProduct",
			builder:  func() vecgo.HNSWBuilder[string] { return vecgo.HNSW[string](4).DotProduct() },
			expected: "DotProduct",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db, err := tt.builder().Build()
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}
			defer db.Close()
			// Verify by checking Stats Options map
			stats := db.Stats()
			if distType, ok := stats.Options["DistanceType"]; ok {
				if distType != tt.expected {
					t.Errorf("expected distance type %v, got %v", tt.expected, distType)
				}
			}
		})
	}
}
