package engine

import (
	"fmt"
	"os"
	"runtime/debug"
)

// GoSafe runs a function in a goroutine and recovers from panics.
// It logs the panic and stack trace instead of crashing the process.
func GoSafe(fn func()) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				// In a real system, this would go to a structured logger.
				// For now, we print to stderr to ensure visibility.
				fmt.Fprintf(os.Stderr, "PANIC RECOVERED in background task: %v\n%s\n", r, debug.Stack())
			}
		}()
		fn()
	}()
}
