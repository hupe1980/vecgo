package diskann

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/index"
)

// TestMutableIndex tests incremental insert/delete/update operations on a mutable DiskANN index.
func TestMutableIndex(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-mutable-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "mutable-index")

	// Create mutable index
	idx, err := New(32, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            16,
		L:            30,
		Alpha:        1.2,
		PQSubvectors: 4,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Test initial state
	if idx.Count() != 0 {
		t.Errorf("Expected count=0, got %d", idx.Count())
	}

	// Insert vectors one by one
	vectors := make([][]float32, 100)
	ids := make([]uint64, 100)
	for i := 0; i < 100; i++ {
		vectors[i] = make([]float32, 32)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}

		id, err := idx.Insert(ctx, vectors[i])
		if err != nil {
			t.Fatalf("Insert[%d]: %v", i, err)
		}
		ids[i] = id
	}

	// Verify count
	if idx.Count() != 100 {
		t.Errorf("After inserts: expected count=100, got %d", idx.Count())
	}

	// Search for first vector
	results, err := idx.KNNSearch(ctx, vectors[0], 5, nil)
	if err != nil {
		t.Fatalf("KNNSearch: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("KNNSearch returned no results")
	}

	// First result should be the query itself
	if results[0].ID != ids[0] {
		t.Logf("Warning: First result ID=%d, expected %d (distance=%f)", results[0].ID, ids[0], results[0].Distance)
	}

	// Delete some vectors
	for i := 0; i < 10; i++ {
		if err := idx.Delete(ctx, ids[i]); err != nil {
			t.Fatalf("Delete[%d]: %v", i, err)
		}
	}

	// Verify count decreased
	if idx.Count() != 90 {
		t.Errorf("After deletes: expected count=90, got %d", idx.Count())
	}

	// Search should not return deleted vectors
	results, err = idx.KNNSearch(ctx, vectors[0], 10, nil)
	if err != nil {
		t.Fatalf("KNNSearch after delete: %v", err)
	}

	for _, r := range results {
		if r.ID == ids[0] {
			t.Errorf("Search returned deleted vector ID=%d", r.ID)
		}
	}

	// Update a vector
	newVec := make([]float32, 32)
	for j := range newVec {
		newVec[j] = rng.Float32()
	}
	if err := idx.Update(ctx, ids[10], newVec); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// Retrieve updated vector
	retrieved, err := idx.VectorByID(ctx, ids[10])
	if err != nil {
		t.Fatalf("VectorByID: %v", err)
	}

	// Verify vector was updated
	for j, v := range newVec {
		if retrieved[j] != v {
			t.Errorf("VectorByID[%d]: got %f, expected %f", j, retrieved[j], v)
			break
		}
	}

	// Test BatchInsert
	batchVectors := make([][]float32, 20)
	for i := range batchVectors {
		batchVectors[i] = make([]float32, 32)
		for j := range batchVectors[i] {
			batchVectors[i][j] = rng.Float32()
		}
	}

	batchResult := idx.BatchInsert(ctx, batchVectors)
	if len(batchResult.IDs) != 20 {
		t.Errorf("BatchInsert: expected 20 IDs, got %d", len(batchResult.IDs))
	}
	for i, err := range batchResult.Errors {
		if err != nil {
			t.Errorf("BatchInsert[%d]: %v", i, err)
		}
	}

	// Final count should be 90 + 20 = 110
	if idx.Count() != 110 {
		t.Errorf("After BatchInsert: expected count=110, got %d", idx.Count())
	}
}

// TestTransactionalIndexInterface verifies that Index implements TransactionalIndex.
func TestTransactionalIndexInterface(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-txn-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "txn-index")

	idx, err := New(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            8,
		L:            20,
		Alpha:        1.2,
		PQSubvectors: 4,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Verify interface compliance at runtime
	var _ index.TransactionalIndex = idx

	ctx := context.Background()

	// Test ID allocation
	id1 := idx.AllocateID()
	id2 := idx.AllocateID()
	if id1 == id2 {
		t.Error("AllocateID returned duplicate IDs")
	}

	// Test ApplyInsert
	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i)
	}

	if err := idx.ApplyInsert(ctx, id1, vec); err != nil {
		t.Fatalf("ApplyInsert: %v", err)
	}

	// Retrieve vector
	retrieved, err := idx.VectorByID(ctx, id1)
	if err != nil {
		t.Fatalf("VectorByID: %v", err)
	}
	if len(retrieved) != 16 {
		t.Fatalf("Expected 16 dimensions, got %d", len(retrieved))
	}

	// Test ApplyDelete
	if err := idx.ApplyDelete(ctx, id1); err != nil {
		t.Fatalf("ApplyDelete: %v", err)
	}

	// VectorByID should return error for deleted vector
	if _, err := idx.VectorByID(ctx, id1); err == nil {
		t.Error("VectorByID should fail for deleted vector")
	}

	// Test ReleaseID
	idx.ReleaseID(id2)

	// Next AllocateID should reuse released ID
	id3 := idx.AllocateID()
	if id3 != id2 {
		t.Logf("Note: AllocateID returned %d, expected to reuse %d (may allocate new if freelist LIFO)", id3, id2)
	}
}

// TestMutableStats verifies Stats reporting for mutable index.
func TestMutableStats(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-stats-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "stats-index")

	idx, err := New(8, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            4,
		L:            10,
		Alpha:        1.2,
		PQSubvectors: 2,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx := context.Background()

	// Insert vectors
	for i := 0; i < 50; i++ {
		vec := make([]float32, 8)
		for j := range vec {
			vec[j] = float32(i*10 + j)
		}
		if _, err := idx.Insert(ctx, vec); err != nil {
			t.Fatalf("Insert[%d]: %v", i, err)
		}
	}

	// Delete some
	for i := uint64(0); i < 5; i++ {
		if err := idx.Delete(ctx, i); err != nil {
			t.Fatalf("Delete[%d]: %v", i, err)
		}
	}

	stats := idx.Stats()

	// Verify mode
	if stats.Options["Mode"] != "mutable" {
		t.Errorf("Expected Mode=mutable, got %s", stats.Options["Mode"])
	}

	// Verify counts (50 - 5 = 45)
	if stats.Storage["VectorCount"] != "45" {
		t.Errorf("Expected VectorCount=45, got %s", stats.Storage["VectorCount"])
	}

	// Verify deletion stats present
	if stats.Storage["DeletedCount"] != "5" {
		t.Errorf("Expected DeletedCount=5, got %s", stats.Storage["DeletedCount"])
	}
	if stats.Storage["TotalCount"] != "50" {
		t.Errorf("Expected TotalCount=50, got %s", stats.Storage["TotalCount"])
	}
}

// TestDimensionMismatch verifies error handling for dimension mismatches.
func TestDimensionMismatch(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-dim-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "dim-index")

	idx, err := New(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            8,
		L:            20,
		Alpha:        1.2,
		PQSubvectors: 4,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx := context.Background()

	// Try to insert wrong dimension
	wrongVec := make([]float32, 32)
	if _, err := idx.Insert(ctx, wrongVec); err == nil {
		t.Error("Insert should fail for wrong dimension")
	}

	// Correct dimension should work
	correctVec := make([]float32, 16)
	for i := range correctVec {
		correctVec[i] = float32(i)
	}
	id, err := idx.Insert(ctx, correctVec)
	if err != nil {
		t.Fatalf("Insert with correct dimension failed: %v", err)
	}

	// Try to update with wrong dimension
	if err := idx.Update(ctx, id, wrongVec); err == nil {
		t.Error("Update should fail for wrong dimension")
	}

	// Search with wrong dimension
	if _, err := idx.KNNSearch(ctx, wrongVec, 1, nil); err == nil {
		t.Error("KNNSearch should fail for wrong dimension")
	}
}

// TestCompaction verifies manual compaction removes deleted vectors and rebuilds the index.
func TestCompaction(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-compact-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "compact-index")

	// Create index with auto-compaction disabled
	idx, err := New(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:                    8,
		L:                    20,
		Alpha:                1.2,
		PQSubvectors:         4,
		PQCentroids:          256,
		EnableAutoCompaction: false, // Manual compaction only
		CompactionThreshold:  0.2,   // 20% deletion triggers compaction
		CompactionMinVectors: 50,    // Low threshold for testing
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Insert 100 vectors
	vectors := make([][]float32, 100)
	ids := make([]uint64, 100)
	for i := 0; i < 100; i++ {
		vectors[i] = make([]float32, 16)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
		ids[i], err = idx.Insert(ctx, vectors[i])
		if err != nil {
			t.Fatalf("Insert[%d]: %v", i, err)
		}
	}

	if idx.Count() != 100 {
		t.Errorf("Expected count=100, got %d", idx.Count())
	}

	// Delete 30 vectors (30% deletion rate)
	for i := 0; i < 30; i++ {
		if err := idx.Delete(ctx, ids[i]); err != nil {
			t.Fatalf("Delete[%d]: %v", i, err)
		}
	}

	if idx.Count() != 70 {
		t.Errorf("After delete: expected count=70, got %d", idx.Count())
	}

	// Compaction should be needed
	if !idx.ShouldCompact() {
		t.Error("ShouldCompact should return true with 30% deletion")
	}

	// Perform compaction
	if err := idx.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	// Verify count unchanged (still 70 live vectors)
	if idx.Count() != 70 {
		t.Errorf("After compact: expected count=70, got %d", idx.Count())
	}

	// Verify no deleted vectors remain
	if idx.ShouldCompact() {
		t.Error("ShouldCompact should return false after compaction")
	}

	// Verify compaction stats
	stats := idx.CompactionStats()
	if stats.TotalCompactions != 1 {
		t.Errorf("Expected TotalCompactions=1, got %d", stats.TotalCompactions)
	}
	if stats.LastVectorsRemoved != 30 {
		t.Errorf("Expected LastVectorsRemoved=30, got %d", stats.LastVectorsRemoved)
	}

	// Search should still work (IDs are remapped after compaction)
	results, err := idx.KNNSearch(ctx, vectors[50], 5, nil)
	if err != nil {
		t.Fatalf("KNNSearch after compact: %v", err)
	}
	if len(results) == 0 {
		t.Error("KNNSearch returned no results after compaction")
	}

	// All returned IDs should be valid (< 70 after compaction)
	for _, r := range results {
		if r.ID >= 70 {
			t.Errorf("Search returned invalid ID=%d (max should be 69)", r.ID)
		}
	}
}

// TestAutoCompaction verifies background compaction triggers automatically.
func TestAutoCompaction(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-auto-compact-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "auto-compact-index")

	// Create index with aggressive auto-compaction settings
	idx, err := New(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:                    8,
		L:                    20,
		Alpha:                1.2,
		PQSubvectors:         4,
		PQCentroids:          256,
		EnableAutoCompaction: true,
		CompactionThreshold:  0.2, // 20% deletion
		CompactionInterval:   1,   // Check every 1 second
		CompactionMinVectors: 10,  // Low threshold for testing
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Insert 50 vectors
	for i := 0; i < 50; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		if _, err := idx.Insert(ctx, vec); err != nil {
			t.Fatalf("Insert[%d]: %v", i, err)
		}
	}

	// Delete 15 vectors (30% deletion rate - above threshold)
	for i := uint64(0); i < 15; i++ {
		if err := idx.Delete(ctx, i); err != nil {
			t.Fatalf("Delete[%d]: %v", i, err)
		}
	}

	// Wait for background compaction (up to 5 seconds)
	for attempt := 0; attempt < 50; attempt++ {
		time.Sleep(100 * time.Millisecond)
		stats := idx.CompactionStats()
		if stats.TotalCompactions > 0 {
			t.Logf("Auto-compaction triggered after ~%d ms", attempt*100)
			return
		}
	}

	t.Error("Auto-compaction did not trigger within 5 seconds")
}

// TestCompactionConcurrency verifies that concurrent operations work during compaction.
func TestCompactionConcurrency(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-concurrent-compact-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "concurrent-compact-index")

	idx, err := New(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:                    8,
		L:                    20,
		Alpha:                1.2,
		PQSubvectors:         4,
		PQCentroids:          256,
		EnableAutoCompaction: false,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Insert vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		if _, err := idx.Insert(ctx, vec); err != nil {
			t.Fatalf("Insert[%d]: %v", i, err)
		}
	}

	// Delete half
	for i := uint64(0); i < 50; i++ {
		if err := idx.Delete(ctx, i); err != nil {
			t.Fatalf("Delete[%d]: %v", i, err)
		}
	}

	// Try concurrent compaction (should fail)
	done := make(chan error, 2)
	start := make(chan struct{})

	go func() {
		<-start
		done <- idx.Compact(ctx)
	}()
	go func() {
		<-start
		done <- idx.Compact(ctx)
	}()

	// Start both goroutines simultaneously
	close(start)

	// One should succeed, one should fail
	err1 := <-done
	err2 := <-done

	successCount := 0
	if err1 == nil {
		successCount++
	}
	if err2 == nil {
		successCount++
	}

	if successCount != 1 {
		t.Errorf("Expected exactly 1 successful compaction, got %d (err1=%v, err2=%v)", successCount, err1, err2)
	}
}

// TestCompactionReadOnly verifies that read-only indexes cannot be compacted.
func TestCompactionReadOnly(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "diskann-readonly-compact-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "readonly-index")

	// Build index with Builder
	builder, err := NewBuilder(16, index.DistanceTypeSquaredL2, indexPath, &Options{
		R:            8,
		L:            20,
		Alpha:        1.2,
		PQSubvectors: 4,
		PQCentroids:  256,
	})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 50; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		if _, err := builder.Add(vec); err != nil {
			t.Fatalf("Add[%d]: %v", i, err)
		}
	}

	ctx := context.Background()
	if err := builder.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Open as read-only
	idx, err := Open(indexPath, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	// Compaction should fail
	if err := idx.Compact(ctx); err == nil {
		t.Error("Compact should fail on read-only index")
	}

	// ShouldCompact should return false
	if idx.ShouldCompact() {
		t.Error("ShouldCompact should return false for read-only index")
	}
}
