package diskann

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/index"
)

func TestBuilder(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test-index")

	// Create builder
	builder, err := NewBuilder(64, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            32,
		L:            50,
		Alpha:        1.2,
		PQSubvectors: 8,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	// Add vectors
	rng := rand.New(rand.NewSource(42))
	n := 1000
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float32, 64)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
	}

	ids, err := builder.AddBatch(vectors)
	if err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if len(ids) != n {
		t.Errorf("AddBatch returned %d ids, expected %d", len(ids), n)
	}

	// Build index
	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Verify files exist
	for _, filename := range []string{MetaFilename, GraphFilename, PQCodesFilename, VectorsFilename} {
		path := filepath.Join(indexPath, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Expected file %s to exist", filename)
		}
	}
}

func TestOpenAndSearch(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test-index")

	// Build index
	builder, err := NewBuilder(64, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            32,
		L:            50,
		Alpha:        1.2,
		PQSubvectors: 8,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	n := 500
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float32, 64)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
	}

	if _, err := builder.AddBatch(vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Open index
	idx, err := Open(indexPath, &Options{
		BeamWidth: 4,
		RerankK:   50,
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	// Verify basic properties
	if idx.Dimension() != 64 {
		t.Errorf("Dimension: got %d, expected 64", idx.Dimension())
	}
	if idx.Count() != n {
		t.Errorf("Count: got %d, expected %d", idx.Count(), n)
	}

	// Search
	query := vectors[0] // Search for first vector
	results, err := idx.KNNSearch(ctx, query, 10, nil)
	if err != nil {
		t.Fatalf("KNNSearch: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("KNNSearch returned empty results")
	}

	// First result should be the query itself (or very close)
	if results[0].ID != 0 {
		// It's okay if it's not ID 0, but distance should be very small
		if results[0].Distance > 1.0 {
			t.Errorf("Expected first result to be very close, got distance %f", results[0].Distance)
		}
	}

	// Results should be sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("Results not sorted: result[%d].Distance=%f < result[%d].Distance=%f",
				i, results[i].Distance, i-1, results[i-1].Distance)
		}
	}
}

func TestSearchStream(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test-index")

	// Build small index
	builder, err := NewBuilder(32, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            16,
		L:            30,
		Alpha:        1.2,
		PQSubvectors: 4,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	n := 200
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float32, 32)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
	}

	if _, err := builder.AddBatch(vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Open and search
	idx, err := Open(indexPath, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	// Stream search
	query := vectors[0]
	count := 0
	for result, err := range idx.KNNSearchStream(ctx, query, 5, nil) {
		if err != nil {
			t.Fatalf("KNNSearchStream error: %v", err)
		}
		count++
		_ = result
	}

	if count != 5 {
		t.Errorf("Expected 5 results, got %d", count)
	}
}

func TestBruteSearch(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test-index")

	// Build small index
	builder, err := NewBuilder(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            8,
		L:            20,
		Alpha:        1.2,
		PQSubvectors: 2,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	n := 100
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float32, 16)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
	}

	if _, err := builder.AddBatch(vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Open and search
	idx, err := Open(indexPath, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	// Brute search should return exact results
	query := vectors[0]
	results, err := idx.BruteSearch(ctx, query, 5, nil)
	if err != nil {
		t.Fatalf("BruteSearch: %v", err)
	}

	if len(results) != 5 {
		t.Errorf("Expected 5 results, got %d", len(results))
	}

	// First result should be the query itself
	if results[0].ID != 0 {
		t.Errorf("Expected first result ID=0, got %d", results[0].ID)
	}
	if results[0].Distance > 1e-6 {
		t.Errorf("Expected first result distance≈0, got %f", results[0].Distance)
	}
}

func TestReadOnlyOperations(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test-index")

	// Build minimal index
	builder, err := NewBuilder(8, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            4,
		L:            10,
		Alpha:        1.2,
		PQSubvectors: 2,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if _, err := builder.Add(vec); err != nil {
		t.Fatalf("Add: %v", err)
	}

	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Open index
	idx, err := Open(indexPath, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	// Insert should fail
	if _, err := idx.Insert(ctx, vec); err == nil {
		t.Error("Insert should fail on read-only index")
	}

	// Delete should fail
	if err := idx.Delete(ctx, 0); err == nil {
		t.Error("Delete should fail on read-only index")
	}

	// Update should fail
	if err := idx.Update(ctx, 0, vec); err == nil {
		t.Error("Update should fail on read-only index")
	}

	// BatchInsert should fail
	result := idx.BatchInsert(ctx, [][]float32{vec})
	if len(result.Errors) == 0 || result.Errors[0] == nil {
		t.Error("BatchInsert should fail on read-only index")
	}
}

func TestStats(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test-index")

	builder, err := NewBuilder(32, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            16,
		L:            30,
		Alpha:        1.2,
		PQSubvectors: 4,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		vec := make([]float32, 32)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		if _, err := builder.Add(vec); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	idx, err := Open(indexPath, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	stats := idx.Stats()

	// Verify stats contain expected fields
	if stats.Options["Dimension"] != "32" {
		t.Errorf("Expected Dimension=32, got %s", stats.Options["Dimension"])
	}
	if stats.Storage["VectorCount"] != "100" {
		t.Errorf("Expected VectorCount=100, got %s", stats.Storage["VectorCount"])
	}
}

func TestFileFormat(t *testing.T) {
	// Test header serialization
	header := FileHeader{
		Magic:        FormatMagic,
		Version:      FormatVersion,
		Flags:        FlagPQEnabled,
		Dimension:    128,
		Count:        1000,
		DistanceType: uint32(index.DistanceTypeSquaredL2),
		R:            64,
		L:            100,
		Alpha:        1200,
		PQSubvectors: 8,
		PQCentroids:  256,
	}

	// Write and read back
	tmpFile, err := os.CreateTemp("", "header-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	if _, err := header.WriteTo(tmpFile); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}

	// Seek back and read
	if _, err := tmpFile.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	var loaded FileHeader
	if _, err := loaded.ReadFrom(tmpFile); err != nil {
		t.Fatalf("ReadFrom: %v", err)
	}

	if err := loaded.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	// Compare fields
	if loaded.Magic != header.Magic {
		t.Errorf("Magic: got 0x%X, expected 0x%X", loaded.Magic, header.Magic)
	}
	if loaded.Dimension != header.Dimension {
		t.Errorf("Dimension: got %d, expected %d", loaded.Dimension, header.Dimension)
	}
	if loaded.Count != header.Count {
		t.Errorf("Count: got %d, expected %d", loaded.Count, header.Count)
	}
	if loaded.AlphaFloat() != 1.2 {
		t.Errorf("Alpha: got %f, expected 1.2", loaded.AlphaFloat())
	}
}
