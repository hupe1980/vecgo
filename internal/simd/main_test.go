package simd

import (
	"fmt"
	"os"
	"runtime"
	"testing"
)

// TestMain runs before all tests and prints ISA diagnostic information.
// This helps CI identify which SIMD implementation is actually being used.
func TestMain(m *testing.M) {
	// Print ISA diagnostic information
	fmt.Printf("=== SIMD ISA Diagnostics ===\n")
	fmt.Printf("GOOS=%s GOARCH=%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("VECGO_SIMD=%q\n", os.Getenv("VECGO_SIMD"))
	fmt.Printf("Active ISA: %s\n", ActiveISA())
	fmt.Printf("Override: %v\n", IsOverridden())
	fmt.Printf("CPU Features:\n")

	switch runtime.GOARCH {
	case "arm64":
		fmt.Printf("  ASIMD (NEON): %v\n", HasASIMD())
		fmt.Printf("  SVE2: %v\n", HasSVE2())
	case "amd64":
		fmt.Printf("  AVX2+FMA: %v\n", HasAVX2())
		fmt.Printf("  AVX-512 (F+BW): %v\n", HasAVX512())
	}

	fmt.Printf("============================\n\n")

	// Run tests
	os.Exit(m.Run())
}
