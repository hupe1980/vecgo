package vecgo

import (
	"fmt"
	"sync"
)

// closeOnce ensures Close() is idempotent (can be called multiple times safely)
var closeOnceMap sync.Map // map[*Vecgo[T]]*sync.Once

// Close releases resources held by this Vecgo instance.
//
// This method is idempotent - calling it multiple times is safe.
// It properly shuts down all background workers and ensures no goroutine leaks.
//
// Resources closed (in dependency order):
// 1. Coordinator (may have background workers)
// 2. WAL (may have group commit worker)
// 3. Mmap resources (if present)
//
// After the first Close() call returns, the Vecgo instance is no longer usable.
func (vg *Vecgo[T]) Close() error {
	// Get or create sync.Once for this instance
	onceVal, _ := closeOnceMap.LoadOrStore(vg, &sync.Once{})
	once := onceVal.(*sync.Once)

	var closeErr error
	once.Do(func() {
		closeErr = vg.closeInternal()
		// Clean up the sync.Once from the map to avoid memory leak
		closeOnceMap.Delete(vg)
	})
	return closeErr
}

func (vg *Vecgo[T]) closeInternal() error {
	var errs []error

	// Close coordinator first (may have background workers)
	if vg.coordinator != nil {
		if err := vg.coordinator.Close(); err != nil {
			errs = append(errs, fmt.Errorf("coordinator: %w", err))
		}
	}

	// Close mmap resources
	if vg.mmapCloser != nil {
		if err := vg.mmapCloser.Close(); err != nil {
			errs = append(errs, fmt.Errorf("mmap: %w", err))
		}
		vg.mmapCloser = nil
	}

	// Return combined errors
	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}
	return nil
}
