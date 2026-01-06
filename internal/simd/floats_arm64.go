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
		sq8uL2BatchPerDimensionImpl = sq8uL2BatchPerDimensionNEON
		popcountImpl = popcountNEON
		hammingImpl = hammingNEON
	}
}

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

func sq8uL2BatchPerDimensionNEON(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	if len(out) > 0 {
		sq8uL2BatchPerDimensionNeon(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&codes[0]),
			unsafe.Pointer(&mins[0]),
			unsafe.Pointer(&invScales[0]),
			int64(dim),
			int64(len(out)),
			unsafe.Pointer(&out[0]),
		)
	}
}

func popcountNEON(a []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return popcountNeon(unsafe.Pointer(&a[0]), int64(n))
}

func hammingNEON(a, b []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return hammingNeon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(n))
}
