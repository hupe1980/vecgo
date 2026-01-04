package resource

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestController_Memory(t *testing.T) {
	// Test with limit
	c := NewController(Config{MemoryLimitBytes: 100})

	// Acquire 50
	err := c.AcquireMemory(context.Background(), 50)
	require.NoError(t, err)
	assert.Equal(t, int64(50), c.MemoryUsage())

	// Acquire 40
	err = c.AcquireMemory(context.Background(), 40)
	require.NoError(t, err)
	assert.Equal(t, int64(90), c.MemoryUsage())

	// TryAcquire 20 (should fail)
	ok := c.TryAcquireMemory(20)
	assert.False(t, ok)
	assert.Equal(t, int64(90), c.MemoryUsage())

	// Acquire 20 (should block/timeout)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err = c.AcquireMemory(ctx, 20)
	assert.ErrorIs(t, err, context.DeadlineExceeded)

	// Release 50
	c.ReleaseMemory(50)
	assert.Equal(t, int64(40), c.MemoryUsage())

	// Now Acquire 20 should succeed
	err = c.AcquireMemory(context.Background(), 20)
	require.NoError(t, err)
	assert.Equal(t, int64(60), c.MemoryUsage())
}

func TestController_UnlimitedMemory(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 0})

	err := c.AcquireMemory(context.Background(), 1000)
	require.NoError(t, err)
	assert.Equal(t, int64(1000), c.MemoryUsage())

	c.ReleaseMemory(500)
	assert.Equal(t, int64(500), c.MemoryUsage())
}

func TestController_Concurrency(t *testing.T) {
	c := NewController(Config{MaxBackgroundWorkers: 2})

	// Acquire 2
	require.NoError(t, c.AcquireBackground(context.Background()))
	require.NoError(t, c.AcquireBackground(context.Background()))

	// Try 3rd
	assert.False(t, c.TryAcquireBackground())

	// Release 1
	c.ReleaseBackground()

	// Try 3rd again
	assert.True(t, c.TryAcquireBackground())
}
