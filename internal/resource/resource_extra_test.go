package resource

import (
	"bytes"
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestController_MemoryNonBlocking(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 100})

	// Use all memory
	err := c.AcquireMemory(100)
	require.NoError(t, err)
	assert.Equal(t, int64(100), c.MemoryUsage())

	// Attempt to acquire 1 more -> should return error immediately (non-blocking)
	err = c.AcquireMemory(1)
	assert.ErrorIs(t, err, ErrMemoryLimitExceeded)

	// Release 10
	c.ReleaseMemory(10)
	assert.Equal(t, int64(90), c.MemoryUsage())

	// Acquire 5 should succeed now
	err = c.AcquireMemory(5)
	assert.NoError(t, err)
	assert.Equal(t, int64(95), c.MemoryUsage())
}

func TestController_NilChecks(t *testing.T) {
	var c *Controller
	assert.NoError(t, c.AcquireMemory(10))
	c.ReleaseMemory(10) // Should not panic
}

func TestController_Background(t *testing.T) {
	c := NewController(Config{MaxBackgroundWorkers: 1})
	err := c.AcquireBackground(t.Context())
	require.NoError(t, err)

	// Second acquire should fail/timeout
	assert.False(t, c.TryAcquireBackground())

	ctx, cancel := context.WithCancel(t.Context())
	cancel() // Cancel immediately
	err = c.AcquireBackground(ctx)
	assert.Error(t, err)

	c.ReleaseBackground()
	// Now success
	assert.True(t, c.TryAcquireBackground())
}

func TestController_IO(t *testing.T) {
	c := NewController(Config{IOLimitBytesPerSec: 1000}) // 1KB/s

	// Small acquire
	err := c.AcquireIO(t.Context(), 100)
	assert.NoError(t, err)

	// Unlimited
	c2 := NewController(Config{})
	err = c2.AcquireIO(t.Context(), 1000000)
	assert.NoError(t, err)
}

func TestController_InternalChecks(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 10})
	err := c.AcquireMemory(-1)
	assert.NoError(t, err)

	c.ReleaseMemory(-1)
	// nothing happens
}

func TestController_MemoryLimit(t *testing.T) {
	c := NewController(Config{MemoryLimitBytes: 1024})
	assert.Equal(t, int64(1024), c.MemoryLimit())

	c2 := NewController(Config{})
	assert.Equal(t, int64(0), c2.MemoryLimit())

	var c3 *Controller
	assert.Equal(t, int64(0), c3.MemoryLimit())
	assert.Equal(t, int64(0), c3.MemoryUsage())
}

func TestController_NilSafe(t *testing.T) {
	var c *Controller

	// All methods should be nil-safe
	assert.NoError(t, c.AcquireMemory(100))
	c.ReleaseMemory(100)

	assert.NoError(t, c.AcquireBackground(t.Context()))
	assert.True(t, c.TryAcquireBackground())
	c.ReleaseBackground()

	assert.NoError(t, c.AcquireIO(t.Context(), 100))
	assert.True(t, c.TryAcquireIO(100))
}

func TestController_TryAcquireIO(t *testing.T) {
	// With limit
	c := NewController(Config{IOLimitBytesPerSec: 1000})

	// Small acquire should succeed
	ok := c.TryAcquireIO(100)
	assert.True(t, ok)

	// Without limit
	c2 := NewController(Config{})
	ok = c2.TryAcquireIO(1000000)
	assert.True(t, ok)
}

func TestRateLimitedWriter(t *testing.T) {
	c := NewController(Config{IOLimitBytesPerSec: 10000})
	ctx := context.Background()

	var buf bytes.Buffer
	w := NewRateLimitedWriter(ctx, &buf, c)

	n, err := w.Write([]byte("hello"))
	assert.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, "hello", buf.String())
}

func TestRateLimitedWriter_Seek(t *testing.T) {
	c := NewController(Config{})
	ctx := context.Background()

	// With seeker
	f := bytes.NewReader([]byte("hello world"))
	var buf bytes.Buffer
	w := NewRateLimitedWriter(ctx, &buf, c)

	// buf is not a seeker
	_, err := w.Seek(0, 0)
	assert.Error(t, err)

	// Test with a real seeker (using a file-like buffer)
	_ = f // bytes.Reader is a seeker but we wrap a writer
}

func TestRateLimitedReader(t *testing.T) {
	c := NewController(Config{IOLimitBytesPerSec: 10000})
	ctx := context.Background()

	data := bytes.NewReader([]byte("hello world"))
	r := NewRateLimitedReader(ctx, data, c)

	buf := make([]byte, 5)
	n, err := r.Read(buf)
	assert.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, "hello", string(buf))
}

func TestRateLimitedReader_ContextCanceled(t *testing.T) {
	c := NewController(Config{IOLimitBytesPerSec: 1}) // Very slow
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	data := bytes.NewReader([]byte("hello world"))
	r := NewRateLimitedReader(ctx, data, c)

	buf := make([]byte, 1000)
	_, err := r.Read(buf)
	assert.Error(t, err)
}
