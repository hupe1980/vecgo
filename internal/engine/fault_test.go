package engine_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	engine "github.com/hupe1980/vecgo/internal/engine"
	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngine_Flush_DiskFull(t *testing.T) {
	// Setup FaultyFS with no initial limit
	baseFS := fs.LocalFS{}
	faultyFS := fs.NewFaultyFS(baseFS)
	faultyFS.SetError(fmt.Errorf("fake disk full"))

	dir := t.TempDir()

	opts := []engine.Option{
		engine.WithFileSystem(faultyFS),
	}

	e, err := engine.Open(dir, 128, distance.MetricL2, opts...)
	require.NoError(t, err)
	defer e.Close()

	// Insert some data to generate WAL traffic and fill memtable
	vec := make([]float32, 128)
	for i := 0; i < 100; i++ {
		_, err := e.Insert(context.Background(), vec, nil, nil)
		require.NoError(t, err)
	}

	// Capture current written bytes
	writtenBefore := faultyFS.GetWritten()
	t.Logf("Bytes written before flush (WAL): %d", writtenBefore)

	// Set limit to trigger failure during Flush
	// Flush writes segment file + payload + manifest update.
	// We want it to fail partway through.
	// Let's set it to fail after a small amount of additional bytes.
	faultyFS.SetLimit(writtenBefore + 1024) // Allow 1KB more, then fail

	// Trigger Flush
	err = e.Commit(context.Background())

	// Expect error
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fake disk full")

	t.Logf("Flush failed as expected: %v", err)

	// Reset limit to allow cleanup
	faultyFS.SetLimit(0)
}

func TestEngine_Compaction_DiskFull(t *testing.T) {
	// Setup FaultyFS
	baseFS := fs.LocalFS{}
	faultyFS := fs.NewFaultyFS(baseFS)
	faultyFS.SetError(fmt.Errorf("fake disk full during compaction"))

	dir := t.TempDir()

	// Disable auto flush/compaction to control them manually
	opts := []engine.Option{
		engine.WithFileSystem(faultyFS),
		engine.WithFlushConfig(engine.FlushConfig{MaxMemTableSize: 1024 * 1024 * 100}), // Large memtable
	}

	e, err := engine.Open(dir, 128, distance.MetricL2, opts...)
	require.NoError(t, err)
	defer e.Close()

	// 1. Create two segments
	vec := make([]float32, 128)

	// Segment 1
	for i := 0; i < 50; i++ {
		e.Insert(context.Background(), vec, nil, nil)
	}
	require.NoError(t, e.Commit(context.Background()))

	// Segment 2
	for i := 0; i < 50; i++ {
		e.Insert(context.Background(), vec, nil, nil)
	}
	require.NoError(t, e.Commit(context.Background()))

	// Measure written so far
	writtenBefore := faultyFS.GetWritten()
	t.Logf("Bytes written before compaction: %d", writtenBefore)

	// Set limit to fail during compaction
	faultyFS.SetLimit(writtenBefore + 500) // Fail quickly

	// Trigger Compaction
	infos := e.SegmentInfo()
	var segIDs []model.SegmentID
	for _, info := range infos {
		segIDs = append(segIDs, info.ID)
	}
	require.Len(t, segIDs, 2)

	err = e.Compact(segIDs, 1)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fake disk full")

	t.Logf("Compaction failed as expected: %v", err)

	// Reset limit
	faultyFS.SetLimit(0)
}

func TestFault_CorruptSegmentHeader(t *testing.T) {
	dir := t.TempDir()
	e, err := engine.Open(dir, 128, distance.MetricL2)
	require.NoError(t, err)

	vec := make([]float32, 128)
	_, err = e.Insert(context.Background(), vec, nil, nil)
	require.NoError(t, err)

	// Force flush to create segment file
	err = e.Commit(context.Background())
	require.NoError(t, err)
	e.Close()

	// Find segment file
	var segFile string
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if !info.IsDir() && filepath.Ext(path) == ".bin" && strings.HasPrefix(filepath.Base(path), "segment_") {
			segFile = path
			return fmt.Errorf("found")
		}
		return nil
	})

	if segFile == "" {
		// If fails, maybe extension is different.
		// But .bin is standard for vecgo flat segments usually.
		// Let's list files just in case it fails, for debugging
		entries, _ := os.ReadDir(dir)
		for _, en := range entries {
			t.Logf("File: %s", en.Name())
		}
		// Try finding *anything* that looks like a segment
		// segments/ folder? tables/ folder?
		// Walk finds everything.
	}
	require.NotEmpty(t, segFile, "Segment file not found")

	// Corrupt header
	f, err := os.OpenFile(segFile, os.O_RDWR, 0644)
	require.NoError(t, err)
	_, err = f.WriteAt([]byte{0xDE, 0xAD, 0xBE, 0xEF}, 0)
	require.NoError(t, err)
	f.Close()

	// Open should fail
	e2, err := engine.Open(dir, 128, distance.MetricL2)
	assert.Error(t, err)
	if e2 != nil {
		e2.Close()
	}
}

func TestFault_ConcurrentClose(t *testing.T) {
	dir := t.TempDir()
	e, err := engine.Open(dir, 128, distance.MetricL2)
	require.NoError(t, err)

	var wg sync.WaitGroup
	start := make(chan struct{})

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			<-start
			vec := make([]float32, 128)
			for j := 0; j < 100; j++ {
				_, err := e.Insert(context.Background(), vec, nil, nil)
				if err != nil {
					if err == engine.ErrClosed {
						return
					}
				}
				time.Sleep(time.Millisecond)
			}
		}(i)
	}

	close(start)
	time.Sleep(50 * time.Millisecond)
	e.Close()
	wg.Wait()
}
