package vecgo_test

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper function to create a random-ish vector
func randomVector(dim int, seed int) []float32 {
	vec := make([]float32, dim)
	for j := range vec {
		vec[j] = float32(seed*j) * 0.01
	}
	return vec
}

// TestNoGoroutineLeaks verifies that all background workers (WAL group commit,
// DiskANN compaction) are properly stopped when Close() is called.
//
// This test ensures:
// 1. WAL group commit worker terminates cleanly
// 2. DiskANN compaction worker terminates cleanly
// 3. Sharded coordinators clean up all shard workers
// 4. No goroutines are leaked after Close()
func TestNoGoroutineLeaks(t *testing.T) {
	tests := []struct {
		name     string
		setupDB  func(t *testing.T) *vecgo.Vecgo[string]
		maxLeaks int // Allow small variance (runtime background goroutines)
	}{
		{
			name: "HNSW with WAL GroupCommit",
			setupDB: func(t *testing.T) *vecgo.Vecgo[string] {
				tmpDir := t.TempDir()
				db, err := vecgo.HNSW[string](128).
					SquaredL2().
					M(16).
					EFConstruction(100).
					WAL(tmpDir, func(o *wal.Options) {
						o.DurabilityMode = wal.DurabilityGroupCommit
						o.GroupCommitInterval = 10 * time.Millisecond
						o.GroupCommitMaxOps = 100
					}).
					Build()
				require.NoError(t, err)
				return db
			},
			maxLeaks: 2,
		},
		{
			name: "HNSW Sharded (4 shards) with WAL",
			setupDB: func(t *testing.T) *vecgo.Vecgo[string] {
				tmpDir := t.TempDir()
				db, err := vecgo.HNSW[string](128).
					SquaredL2().
					M(16).
					EFConstruction(100).
					Shards(4).
					WAL(tmpDir, func(o *wal.Options) {
						o.DurabilityMode = wal.DurabilityGroupCommit
						o.GroupCommitInterval = 10 * time.Millisecond
					}).
					Build()
				require.NoError(t, err)
				return db
			},
			maxLeaks: 2,
		},
		{
			name: "DiskANN with auto-compaction",
			setupDB: func(t *testing.T) *vecgo.Vecgo[string] {
				tmpDir := t.TempDir()
				db, err := vecgo.DiskANN[string](tmpDir, 128).
					SquaredL2().
					R(32).
					L(50).
					EnableAutoCompaction(true).
					CompactionInterval(100). // milliseconds as int
					Build()
				require.NoError(t, err)
				return db
			},
			maxLeaks: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Force GC to clean up any lingering goroutines from previous tests
			runtime.GC()
			time.Sleep(50 * time.Millisecond)

			initial := runtime.NumGoroutine()
			t.Logf("Initial goroutines: %d", initial)

			// Create and populate database
			db := tt.setupDB(t)

			// Insert data to ensure workers are active
			ctx := context.Background()
			for i := 0; i < 50; i++ {
				_, err := db.Insert(ctx, vecgo.VectorWithData[string]{
					Vector: randomVector(128, i),
					Data:   fmt.Sprintf("doc-%d", i),
					Metadata: metadata.Metadata{
						"category": metadata.String(fmt.Sprintf("cat-%d", i%5)),
					},
				})
				require.NoError(t, err)
			}

			// Perform some searches to exercise the index
			query := randomVector(128, 999)
			_, err := db.Search(query).KNN(10).Execute(ctx)
			require.NoError(t, err)

			afterInsert := runtime.NumGoroutine()
			t.Logf("After inserts: %d goroutines (+%d)", afterInsert, afterInsert-initial)

			// Wait for background workers to start (WAL ticker, compaction)
			time.Sleep(150 * time.Millisecond)

			beforeClose := runtime.NumGoroutine()
			t.Logf("Before close: %d goroutines", beforeClose)

			// Close the database
			err = db.Close()
			require.NoError(t, err)

			// Wait for background workers to fully shut down.
			// This reduces flakiness from asynchronous shutdown timing without weakening
			// leak detection semantics: we still fail if the goroutines don't go away.
			deadline := time.Now().Add(2 * time.Second)
			var final int
			var leaked int
			for {
				runtime.GC()
				time.Sleep(50 * time.Millisecond)

				final = runtime.NumGoroutine()
				leaked = final - initial
				if leaked <= tt.maxLeaks || time.Now().After(deadline) {
					break
				}
			}

			t.Logf("Final goroutines: %d (leaked: %d)", final, leaked)

			// Verify no significant goroutine leaks
			if leaked > tt.maxLeaks {
				t.Errorf("Goroutine leak detected: started with %d, ended with %d (leaked: %d, max allowed: %d)",
					initial, final, leaked, tt.maxLeaks)

				// Print goroutine stack traces for debugging
				buf := make([]byte, 1<<20)
				stackSize := runtime.Stack(buf, true)
				t.Logf("Goroutine stacks:\n%s", buf[:stackSize])
			}
		})
	}
}

// TestCloseIdempotent verifies that calling Close() multiple times is safe.
func TestCloseIdempotent(t *testing.T) {
	tmpDir := t.TempDir()
	db, err := vecgo.HNSW[string](128).
		SquaredL2().
		WAL(tmpDir, func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilityGroupCommit
			o.GroupCommitInterval = 10 * time.Millisecond
		}).
		Build()
	require.NoError(t, err)

	// Insert some data
	ctx := context.Background()
	for i := 0; i < 10; i++ {
		_, err := db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: randomVector(128, i),
			Data:   fmt.Sprintf("doc-%d", i),
		})
		require.NoError(t, err)
	}

	// Close multiple times should not panic or error
	err1 := db.Close()
	err2 := db.Close()
	err3 := db.Close()

	assert.NoError(t, err1, "First close should succeed")
	assert.NoError(t, err2, "Second close should be idempotent")
	assert.NoError(t, err3, "Third close should be idempotent")
}

// TestCloseWithActiveOperations verifies graceful shutdown during active operations.
func TestCloseWithActiveOperations(t *testing.T) {
	tmpDir := t.TempDir()
	db, err := vecgo.HNSW[string](128).
		SquaredL2().
		Shards(4).
		WAL(tmpDir, func(o *wal.Options) {
			o.DurabilityMode = wal.DurabilityGroupCommit
			o.GroupCommitInterval = 5 * time.Millisecond
		}).
		Build()
	require.NoError(t, err)

	ctx := context.Background()

	// Start concurrent inserts
	done := make(chan bool)
	go func() {
		for i := 0; i < 100; i++ {
			db.Insert(ctx, vecgo.VectorWithData[string]{
				Vector: randomVector(128, i),
				Data:   fmt.Sprintf("doc-%d", i),
			})
			time.Sleep(1 * time.Millisecond)
		}
		done <- true
	}()

	// Let some inserts happen
	time.Sleep(50 * time.Millisecond)

	// Close while operations are active
	err = db.Close()
	assert.NoError(t, err, "Close should succeed even with active operations")

	// Wait for goroutine to finish
	<-done
}
