package resource

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestController_MemoryBlocking(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 100})

	// Use all memory
	err := c.AcquireMemory(context.Background(), 100)
	require.NoError(t, err)
	assert.Equal(t, int64(100), c.MemoryUsage())

	// Attempt to acquire 1 more -> should block
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err = c.AcquireMemory(ctx, 1)
	assert.ErrorIs(t, err, context.DeadlineExceeded)

	// TryAcquire should fail instantly
	ok := c.TryAcquireMemory(1)
	assert.False(t, ok)

	// Release 10
	c.ReleaseMemory(10)
	assert.Equal(t, int64(90), c.MemoryUsage())

	// TryAcquire 5 should succeed now
	ok = c.TryAcquireMemory(5)
	assert.True(t, ok)
	assert.Equal(t, int64(95), c.MemoryUsage())
}

func TestController_NilChecks(t *testing.T) {
	var c *Controller
	assert.NoError(t, c.AcquireMemory(context.Background(), 10))
	assert.True(t, c.TryAcquireMemory(10))
	c.ReleaseMemory(10) // Should not panic
}

func TestController_Background(t *testing.T) {
	c := NewController(Config{MaxBackgroundWorkers: 1})
	err := c.AcquireBackground(context.Background())
	require.NoError(t, err)

	// Second acquire should fail/timeout
	assert.False(t, c.TryAcquireBackground())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err = c.AcquireBackground(ctx)
	assert.Error(t, err)

	c.ReleaseBackground()
	// Now success
	assert.True(t, c.TryAcquireBackground())
}

func TestController_IO(t *testing.T) {
	c := NewController(Config{IOLimitBytesPerSec: 1000}) // 1KB/s
	ctx := context.Background()

	// Small acquire
	err := c.AcquireIO(ctx, 100)
	assert.NoError(t, err)

	// Unlimited
	c2 := NewController(Config{})
	err = c2.AcquireIO(ctx, 1000000)
	assert.NoError(t, err)
}

func TestController_InternalChecks(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 10})
	err := c.AcquireMemory(context.Background(), -1)
	assert.NoError(t, err)

	ok := c.TryAcquireMemory(-1)
	assert.True(t, ok)

	c.ReleaseMemory(-1)
	// nothing happens
}
