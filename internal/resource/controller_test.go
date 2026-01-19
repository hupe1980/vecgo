package resource

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestController_Memory(t *testing.T) {
	// Test with limit
	c := NewController(Config{MemoryLimitBytes: 100})

	// Acquire 50
	err := c.AcquireMemory(50)
	require.NoError(t, err)
	assert.Equal(t, int64(50), c.MemoryUsage())

	// Acquire 40
	err = c.AcquireMemory(40)
	require.NoError(t, err)
	assert.Equal(t, int64(90), c.MemoryUsage())

	// Acquire 20 (should fail - limit exceeded)
	err = c.AcquireMemory(20)
	assert.ErrorIs(t, err, ErrMemoryLimitExceeded)
	assert.Equal(t, int64(90), c.MemoryUsage())

	// Release 50
	c.ReleaseMemory(50)
	assert.Equal(t, int64(40), c.MemoryUsage())

	// Now Acquire 20 should succeed
	err = c.AcquireMemory(20)
	require.NoError(t, err)
	assert.Equal(t, int64(60), c.MemoryUsage())
}

func TestController_UnlimitedMemory(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 0})

	err := c.AcquireMemory(1000)
	require.NoError(t, err)
	assert.Equal(t, int64(1000), c.MemoryUsage())

	c.ReleaseMemory(500)
	assert.Equal(t, int64(500), c.MemoryUsage())
}

func TestController_Concurrency(t *testing.T) {
	c := NewController(Config{MaxBackgroundWorkers: 2})

	// Acquire 2
	require.NoError(t, c.AcquireBackground(t.Context()))
	require.NoError(t, c.AcquireBackground(t.Context()))

	// Try 3rd
	assert.False(t, c.TryAcquireBackground())

	// Release 1
	c.ReleaseBackground()

	// Try 3rd again
	assert.True(t, c.TryAcquireBackground())
}
