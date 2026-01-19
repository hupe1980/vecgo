// Package resource implements the ResourceController for global limits and governance.
//
// The ResourceController provides centralized management of three resource types:
//
//   - Memory: Track and limit memory usage across the engine (non-blocking, fail-fast)
//   - Concurrency: Limit background worker threads (compaction, etc.)
//   - IO: Rate-limit background IO to avoid starving foreground queries
//
// # Architecture
//
//	┌─────────────────────────────────────────────────────────────┐
//	│                    ResourceController                       │
//	├─────────────────┬─────────────────┬─────────────────────────┤
//	│  Memory Limit   │  Background     │  IO Rate Limiter        │
//	│  (fail-fast)    │  Workers (sem)  │  (token bucket)         │
//	├─────────────────┼─────────────────┼─────────────────────────┤
//	│  AcquireMemory  │  AcquireBack-   │  AcquireIO              │
//	│  (non-blocking) │  ground         │  RateLimitedWriter      │
//	│  ReleaseMemory  │  TryAcquire     │  RateLimitedReader      │
//	│  MemoryUsage    │  Release        │                         │
//	└─────────────────┴─────────────────┴─────────────────────────┘
//
// # Memory Management
//
// Memory tracking uses a weighted semaphore for hard limits and atomic counters
// for usage tracking. AcquireMemory is non-blocking and returns immediately
// with ErrMemoryLimitExceeded if the limit would be exceeded:
//
//	rc := resource.NewController(resource.Config{
//	    MemoryLimitBytes: 1 << 30, // 1GB limit
//	})
//
//	// Non-blocking acquire (returns error immediately if limit exceeded)
//	if err := rc.AcquireMemory(1024*1024); err != nil {
//	    // ErrMemoryLimitExceeded - caller decides retry/backoff
//	}
//	defer rc.ReleaseMemory(1024*1024)
//
// # Background Worker Limits
//
// Limits concurrent background operations (compaction, index building):
//
//	rc := resource.NewController(resource.Config{
//	    MaxBackgroundWorkers: 4,
//	})
//
//	if err := rc.AcquireBackground(ctx); err != nil {
//	    return err
//	}
//	defer rc.ReleaseBackground()
//
// # IO Rate Limiting
//
// Token bucket rate limiter for background IO to prevent starving foreground queries:
//
//	rc := resource.NewController(resource.Config{
//	    IOLimitBytesPerSec: 100 * 1024 * 1024, // 100MB/s
//	})
//
//	// Direct acquire
//	if err := rc.AcquireIO(ctx, 4096); err != nil {
//	    return err
//	}
//
//	// Rate-limited writer/reader wrappers
//	writer := resource.NewRateLimitedWriter(ctx, file, rc)
//	reader := resource.NewRateLimitedReader(ctx, file, rc)
//
// # Thread Safety
//
// All Controller methods are safe for concurrent use. The underlying
// implementations use atomic operations and sync primitives.
//
// # Nil Safety
//
// All methods handle nil Controller gracefully - they become no-ops.
// This allows optional resource limiting without nil checks everywhere.
package resource
