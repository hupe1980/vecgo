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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFault_CorruptSegmentHeader(t *testing.T) {
	dir := t.TempDir()
	e, err := engine.OpenLocal(context.Background(), dir,
		engine.WithDimension(128),
		engine.WithMetric(distance.MetricL2),
	)
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
	e2, err := engine.OpenLocal(context.Background(), dir,
		engine.WithDimension(128),
		engine.WithMetric(distance.MetricL2),
	)
	assert.Error(t, err)
	if e2 != nil {
		e2.Close()
	}
}

func TestFault_ConcurrentClose(t *testing.T) {
	dir := t.TempDir()
	e, err := engine.OpenLocal(context.Background(), dir,
		engine.WithDimension(128),
		engine.WithMetric(distance.MetricL2),
	)
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
