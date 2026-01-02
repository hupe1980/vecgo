//go:build arm64 && !noasm

package math32

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.ARM64.HasASIMD {
		dotImpl = dotNEON
		squaredL2Impl = squaredL2NEON
		scaleImpl = scaleNEON
		pqAdcImpl = pqAdcNEON
	}
}

//go:noescape
func dotProductNeon(a unsafe.Pointer, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func squaredL2Neon(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func pqAdcLookupNeon(table, codes unsafe.Pointer, m int64, result unsafe.Pointer)

// NOTE: The generated assembly currently expects 3 args laid out as:
// a @ +0, n @ +8, scalar @ +16.
//
//go:noescape
func scaleNeon(a unsafe.Pointer, n int64, scalar unsafe.Pointer)

func dotNEON(a, b []float32) float32 {
	var ret float32

	if len(a) > 0 {
		dotProductNeon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret)) //nolint:gosec // unsafe is required for SIMD
	}

	return ret
}

func squaredL2NEON(a, b []float32) float32 {
	var ret float32

	if len(a) > 0 {
		squaredL2Neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret)) //nolint:gosec // unsafe is required for SIMD
	}

	return ret
}

func scaleNEON(a []float32, scalar float32) {
	if len(a) == 0 {
		return
	}
	s := scalar
	scaleNeon(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&s)) //nolint:gosec // unsafe is required for SIMD
}

func pqAdcNEON(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupNeon(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret)) //nolint:gosec // unsafe is required for SIMD
	}
	return ret
}
