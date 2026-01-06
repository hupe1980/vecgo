package wal

import (
	"path/filepath"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWAL_GroupCommit_Concurrency(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "wal.log")

	opts := Options{Durability: DurabilitySync}
	w, err := Open(nil, path, opts)
	require.NoError(t, err)
	defer w.Close()

	concurrency := 50
	recordsPerGoroutine := 100
	totalRecords := concurrency * recordsPerGoroutine

	var wg sync.WaitGroup
	wg.Add(concurrency)

	// Start concurrent writers
	for i := 0; i < concurrency; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < recordsPerGoroutine; j++ {
				pk := model.PKUint64(uint64(id*recordsPerGoroutine + j))
				rec := &Record{
					LSN:    uint64(id*recordsPerGoroutine + j),
					Type:   RecordTypeUpsert,
					PK:     pk,
					Vector: []float32{1.0, 2.0, 3.0},
				}
				if err := w.Append(rec); err != nil {
					panic(err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Close and reopen to verify data
	require.NoError(t, w.Close())

	w2, err := Open(nil, path, opts)
	require.NoError(t, err)
	defer w2.Close()

	reader, err := w2.Reader()
	require.NoError(t, err)
	defer reader.Close()

	count := 0
	seen := make(map[uint64]bool)
	for {
		rec, err := reader.Next()
		if err != nil {
			break
		}
		count++
		u64, _ := rec.PK.Uint64()
		seen[u64] = true
	}

	assert.Equal(t, totalRecords, count)
	assert.Equal(t, totalRecords, len(seen))
}

func TestWAL_GroupCommit_Sync(t *testing.T) {
	// Verify that Sync() works correctly with group commit
	dir := t.TempDir()
	path := filepath.Join(dir, "wal_sync.log")

	opts := Options{Durability: DurabilitySync}
	w, err := Open(nil, path, opts)
	require.NoError(t, err)
	defer w.Close()

	// Append without sync (internal append does sync, but we want to test explicit Sync too)
	// Actually Append calls Sync internally if DurabilitySync.
	// Let's switch to Async mode, append, then Sync.
	// But GroupCommit logic is only active in Sync mode.

	// In Sync mode, Append waits.
	// Let's just verify Sync() returns successfully.
	err = w.Sync()
	assert.NoError(t, err)

	rec := &Record{
		LSN:    1,
		Type:   RecordTypeUpsert,
		PK:     model.PKUint64(1),
		Vector: []float32{1.0},
	}
	err = w.Append(rec)
	assert.NoError(t, err)

	err = w.Sync()
	assert.NoError(t, err)
}
