package simd

import (
	"os"
	"runtime"
	"strings"
)

// ISA represents a SIMD instruction set architecture.
type ISA uint8

const (
	// Generic represents pure Go implementation (no SIMD).
	Generic ISA = iota
	// NEON represents ARM64 NEON (128-bit SIMD, ASIMD).
	NEON
	// SVE2 represents ARM64 SVE2 (scalable vectors, 128-2048 bit).
	SVE2
	// AVX2 represents x86-64 AVX2 (256-bit SIMD with FMA).
	AVX2
	// AVX512 represents x86-64 AVX-512 (512-bit SIMD).
	AVX512
)

// String returns the string representation of an ISA.
func (i ISA) String() string {
	switch i {
	case Generic:
		return "generic"
	case NEON:
		return "neon"
	case SVE2:
		return "sve2"
	case AVX2:
		return "avx2"
	case AVX512:
		return "avx512"
	default:
		return "unknown"
	}
}

// ParseISA parses a string into an ISA value.
func ParseISA(s string) (ISA, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "generic":
		return Generic, true
	case "neon":
		return NEON, true
	case "sve2":
		return SVE2, true
	case "avx2":
		return AVX2, true
	case "avx512":
		return AVX512, true
	default:
		return Generic, false
	}
}

// Package-level state - initialized once at package init.
// No mutex needed: Go guarantees init() runs before any other code.
var (
	// activeISA is the selected SIMD implementation.
	activeISA ISA

	// hasOverride is true if VECGO_SIMD was set.
	hasOverride bool

	// CPU feature flags (set by platform-specific init)
	hasASIMD    bool // ARM64 NEON
	hasSVE2     bool // ARM64 SVE2
	hasAVX2     bool // x86-64 AVX2 + FMA
	hasAVX512F  bool // x86-64 AVX-512 Foundation
	hasAVX512BW bool // x86-64 AVX-512 Byte/Word
)

// initCapabilities is called from platform-specific init functions
// after CPU features are detected.
func initCapabilities() {
	// Check for environment override
	if override := os.Getenv("VECGO_SIMD"); override != "" {
		if isa, ok := ParseISA(override); ok {
			hasOverride = true
			// Validate the override is available
			if isISAAvailable(isa) {
				activeISA = isa
				return
			}
			// Invalid override - fall through to auto-detection
		}
	}

	// Auto-select best ISA
	activeISA = selectBestISA()
}

// isISAAvailable checks if an ISA is supported on this CPU.
func isISAAvailable(isa ISA) bool {
	switch isa {
	case Generic:
		return true
	case NEON:
		return hasASIMD
	case SVE2:
		return hasSVE2
	case AVX2:
		return hasAVX2
	case AVX512:
		return hasAVX512F && hasAVX512BW
	default:
		return false
	}
}

// selectBestISA chooses the optimal ISA for the current platform.
func selectBestISA() ISA {
	switch runtime.GOARCH {
	case "arm64":
		return selectBestARM64()
	case "amd64":
		return selectBestAMD64()
	default:
		return Generic
	}
}

// selectBestARM64 selects the best ISA for ARM64.
func selectBestARM64() ISA {
	// On macOS (Apple Silicon), NEON is faster than SVE2 because
	// Apple's SVE2 support is emulated. On Linux ARM servers
	// (Graviton, Ampere), SVE2 is native and 2x faster.
	preferNEON := runtime.GOOS == "darwin"

	if hasSVE2 && !preferNEON {
		return SVE2
	}
	if hasASIMD {
		return NEON
	}
	return Generic
}

// selectBestAMD64 selects the best ISA for AMD64.
func selectBestAMD64() ISA {
	// AVX-512 requires both Foundation and BW for our kernels
	if hasAVX512F && hasAVX512BW {
		return AVX512
	}
	if hasAVX2 {
		return AVX2
	}
	return Generic
}

// ActiveISA returns the currently active ISA.
func ActiveISA() ISA {
	return activeISA
}

// IsOverridden returns true if VECGO_SIMD was set.
func IsOverridden() bool {
	return hasOverride
}

// HasASIMD returns true if ARM64 NEON is available.
func HasASIMD() bool {
	return hasASIMD
}

// HasSVE2 returns true if ARM64 SVE2 is available.
func HasSVE2() bool {
	return hasSVE2
}

// HasAVX2 returns true if x86-64 AVX2+FMA is available.
func HasAVX2() bool {
	return hasAVX2
}

// HasAVX512 returns true if x86-64 AVX-512 (F+BW) is available.
func HasAVX512() bool {
	return hasAVX512F && hasAVX512BW
}
