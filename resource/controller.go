package resource

import (
	"context"
	"sync/atomic"

	"golang.org/x/sync/semaphore"
	"golang.org/x/time/rate"
)

// Config holds resource limits.
type Config struct {
	// MemoryLimitBytes is the hard limit for managed memory.
	// If 0, no hard limit is enforced (only tracking).
	MemoryLimitBytes int64

	// MaxBackgroundWorkers is the maximum number of concurrent background jobs.
	// If 0, defaults to 1.
	MaxBackgroundWorkers int64

	// IOLimitBytesPerSec is the maximum IO throughput for background tasks.
	// If 0, unlimited.
	IOLimitBytesPerSec int64
}

// Controller manages global resources (memory, concurrency).
type Controller struct {
	cfg Config

	// Memory
	memSem  *semaphore.Weighted // nil if unlimited
	memUsed atomic.Int64

	// Concurrency
	bgSem *semaphore.Weighted

	// IO
	ioLimiter *rate.Limiter
}

// NewController creates a new resource controller.
func NewController(cfg Config) *Controller {
	if cfg.MaxBackgroundWorkers <= 0 {
		cfg.MaxBackgroundWorkers = 1
	}

	c := &Controller{
		cfg:   cfg,
		bgSem: semaphore.NewWeighted(cfg.MaxBackgroundWorkers),
	}

	if cfg.MemoryLimitBytes > 0 {
		c.memSem = semaphore.NewWeighted(cfg.MemoryLimitBytes)
	}

	if cfg.IOLimitBytesPerSec > 0 {
		c.ioLimiter = rate.NewLimiter(rate.Limit(cfg.IOLimitBytesPerSec), int(cfg.IOLimitBytesPerSec))
	}

	return c
}

// AcquireMemory attempts to reserve memory.
// If a hard limit is configured and usage would exceed it,
// this blocks until memory is available or ctx is canceled.
func (c *Controller) AcquireMemory(ctx context.Context, bytes int64) error {
	if c == nil {
		return nil
	}
	if bytes <= 0 {
		return nil
	}

	if c.memSem != nil {
		if err := c.memSem.Acquire(ctx, bytes); err != nil {
			return err
		}
	}

	c.memUsed.Add(bytes)
	return nil
}

// TryAcquireMemory attempts to reserve memory without blocking.
// Returns true if acquired, false if limit would be exceeded.
func (c *Controller) TryAcquireMemory(bytes int64) bool {
	if c == nil {
		return true
	}
	if bytes <= 0 {
		return true
	}

	if c.memSem != nil {
		if !c.memSem.TryAcquire(bytes) {
			return false
		}
	}

	c.memUsed.Add(bytes)
	return true
}

// ReleaseMemory releases reserved memory.
func (c *Controller) ReleaseMemory(bytes int64) {
	if c == nil {
		return
	}
	if bytes <= 0 {
		return
	}

	if c.memSem != nil {
		c.memSem.Release(bytes)
	}
	c.memUsed.Add(-bytes)
}

// MemoryUsage returns the current memory usage in bytes.
func (c *Controller) MemoryUsage() int64 {
	return c.memUsed.Load()
}

// AcquireBackground attempts to reserve a background worker slot.
// Blocks if all slots are busy.
func (c *Controller) AcquireBackground(ctx context.Context) error {
	return c.bgSem.Acquire(ctx, 1)
}

// ReleaseBackground releases a background worker slot.
func (c *Controller) ReleaseBackground() {
	c.bgSem.Release(1)
}

// AcquireIO waits until the IO limit allows the specified number of bytes.
func (c *Controller) AcquireIO(ctx context.Context, bytes int) error {
	if c.ioLimiter == nil {
		return nil
	}
	return c.ioLimiter.WaitN(ctx, bytes)
}

// TryAcquireBackground attempts to reserve a background worker slot without blocking.
func (c *Controller) TryAcquireBackground() bool {
	return c.bgSem.TryAcquire(1)
}
