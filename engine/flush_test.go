package engine

import (
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAutoFlush(t *testing.T) {
	dir := t.TempDir()
	dim := 4

	// Configure small flush threshold
	flushCfg := FlushConfig{
		MaxMemTableSize: 1024, // 1KB
	}

	e, err := Open(dir, dim, distance.MetricL2, WithFlushConfig(flushCfg))
	require.NoError(t, err)
	defer e.Close()

	// Get initial active ID
	initialSnap := e.current.Load()
	initialID := initialSnap.active.ID()

	// Insert vectors until flush triggers
	// Each vector is 4*4 = 16 bytes.
	// Plus overhead.
	// 1KB should be reached quickly.

	done := make(chan struct{})
	go func() {
		for i := 0; i < 1000; i++ {
			pk := model.PrimaryKey(i)
			vec := []float32{0.1, 0.2, 0.3, 0.4}
			err := e.Insert(pk, vec, nil, nil)
			assert.NoError(t, err)

			// Check if flushed
			snap := e.current.Load()
			if snap.active.ID() > initialID {
				close(done)
				return
			}
			time.Sleep(1 * time.Millisecond)
		}
	}()

	select {
	case <-done:
		// Success
	case <-time.After(5 * time.Second):
		t.Fatal("Flush did not trigger within timeout")
	}

	// Verify persistence
	// Close and reopen
	e.Close()

	e2, err := Open(dir, dim, distance.MetricL2)
	require.NoError(t, err)
	defer e2.Close()

	snap2 := e2.current.Load()
	// Should have at least one immutable segment (L0 flushed to L1)
	assert.NotEmpty(t, snap2.segments)
}
