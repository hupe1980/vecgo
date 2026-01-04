//go:build arm64 && !noasm

package simd

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

		squaredL2BatchImpl = squaredL2BatchNEON
		dotBatchImpl = dotBatchNEON
		f16ToF32Impl = f16ToF32NEON
		sq8L2BatchImpl = sq8L2BatchNEON
		popcountImpl = popcountNEON
		// hammingImpl = hammingNEON   // Fails tests
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

//go:noescape
func squaredL2BatchNeon(query, targets unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func dotBatchNeon(query, targets unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func f16ToF32Neon(in, out unsafe.Pointer, n int64)

//go:noescape
func sq8L2BatchNeon(query, codes, scales, biases unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func popcountNeon(a unsafe.Pointer, n int) int64

//go:noescape
func hammingNeon(a, b unsafe.Pointer, n int) int64

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
	if len(a) > 0 {
		scaleNeon(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&scalar))
	}
}

func pqAdcNEON(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupNeon(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2BatchNEON(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		squaredL2BatchNeon(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func dotBatchNEON(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		dotBatchNeon(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func f16ToF32NEON(in []uint16, out []float32) {
	if len(in) > 0 {
		f16ToF32Neon(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), int64(len(in)))
	}
}

func sq8L2BatchNEON(query []float32, codes []int8, scales []float32, biases []float32, dim int, out []float32) {
	if len(out) > 0 {
		sq8L2BatchNeon(unsafe.Pointer(&query[0]), unsafe.Pointer(&codes[0]), unsafe.Pointer(&scales[0]), unsafe.Pointer(&biases[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func popcountNEON(a []byte) int64 {
	if len(a) == 0 {
		return 0
	}
	return popcountNeon(unsafe.Pointer(&a[0]), len(a))
}

func hammingNEON(a, b []byte) int64 {
	if len(a) == 0 {
		return 0
	}
	return hammingNeon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), len(a))
}
