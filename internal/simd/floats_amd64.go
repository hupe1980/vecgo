//go:build amd64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.X86.HasAVX512F { // Use HasAVX512F as baseline for AVX-512
		dotImpl = dotAVX512
		squaredL2Impl = squaredL2AVX512
		scaleImpl = scaleAVX512
		pqAdcImpl = pqAdcAVX512

		squaredL2BatchImpl = squaredL2BatchAVX512
		dotBatchImpl = dotBatchAVX512
		f16ToF32Impl = f16ToF32AVX512
		sq8L2BatchImpl = sq8L2BatchAVX512
		sq8uL2BatchPerDimensionImpl = sq8uL2BatchPerDimensionAVX512
		popcountImpl = popcountAVX512
		hammingImpl = hammingAVX512
		return
	}
	if cpu.X86.HasAVX {
		dotImpl = dotAVX
		squaredL2Impl = squaredL2AVX
		scaleImpl = scaleAVX
		pqAdcImpl = pqAdcAVX

		squaredL2BatchImpl = squaredL2BatchAVX
		dotBatchImpl = dotBatchAVX
		f16ToF32Impl = f16ToF32AVX
		sq8L2BatchImpl = sq8L2BatchAVX
		sq8uL2BatchPerDimensionImpl = sq8uL2BatchPerDimensionAVX
		popcountImpl = popcountAVX
		hammingImpl = hammingAVX
	}
}

func dotAVX(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		dotProductAvx(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func dotAVX512(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		dotProductAvx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2AVX(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		squaredL2Avx(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2AVX512(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		squaredL2Avx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func pqAdcAVX(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupAvx(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&codes[0]),
			int64(m),
			unsafe.Pointer(&ret),
			unsafe.Pointer(&pqAdcAVXOffsets[0]),
		)
	}
	return ret
}

func pqAdcAVX512(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupAvx512(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&codes[0]),
			int64(m),
			unsafe.Pointer(&ret),
			unsafe.Pointer(&pqAdcAVX512Offsets[0]),
		)
	}
	return ret
}

var pqAdcAVXOffsets = [8]uint32{0, 256, 512, 768, 1024, 1280, 1536, 1792}

var pqAdcAVX512Offsets = [16]uint32{0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840}

func squaredL2BatchAVX(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		squaredL2BatchAvx(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func squaredL2BatchAVX512(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		squaredL2BatchAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func dotBatchAVX(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		dotBatchAvx(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func dotBatchAVX512(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		dotBatchAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func f16ToF32AVX(in []uint16, out []float32) {
	if len(in) > 0 {
		f16ToF32Avx(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), int64(len(in)))
	}
}

func f16ToF32AVX512(in []uint16, out []float32) {
	if len(in) > 0 {
		f16ToF32Avx512(unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]), int64(len(in)))
	}
}

func sq8L2BatchAVX(query []float32, codes []int8, scales []float32, biases []float32, dim int, out []float32) {
	if len(out) > 0 {
		sq8L2BatchAvx(unsafe.Pointer(&query[0]), unsafe.Pointer(&codes[0]), unsafe.Pointer(&scales[0]), unsafe.Pointer(&biases[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func scaleAVX(a []float32, scalar float32) {
	if len(a) > 0 {
		scaleAvx(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&scalar))
	}
}

func scaleAVX512(a []float32, scalar float32) {
	if len(a) > 0 {
		scaleAvx512(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&scalar))
	}
}

func sq8L2BatchAVX512(query []float32, codes []int8, scales []float32, biases []float32, dim int, out []float32) {
	if len(out) > 0 {
		sq8L2BatchAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&codes[0]), unsafe.Pointer(&scales[0]), unsafe.Pointer(&biases[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func sq8uL2BatchPerDimensionAVX(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	if len(query) > 0 {
		sq8uL2BatchPerDimensionAvx(unsafe.Pointer(&query[0]), unsafe.Pointer(&codes[0]), unsafe.Pointer(&mins[0]), unsafe.Pointer(&invScales[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func sq8uL2BatchPerDimensionAVX512(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	if len(query) > 0 {
		sq8uL2BatchPerDimensionAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&codes[0]), unsafe.Pointer(&mins[0]), unsafe.Pointer(&invScales[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}
func popcountAVX(a []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}

	return popcountAvx(
		unsafe.Pointer(&a[0]),
		int64(n),
		unsafe.Pointer(&popcountAVXLookup[0]),
		unsafe.Pointer(&popcountAVXLowMask[0]),
	)
}

func popcountAVX512(a []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}

	return popcountAvx512(unsafe.Pointer(&a[0]), int64(n))
}

func hammingAVX(a, b []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}

	return hammingAvx(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		int64(n),
		unsafe.Pointer(&popcountAVXLookup[0]),
		unsafe.Pointer(&popcountAVXLowMask[0]),
	)
}

// popcountAVXLookup matches the byte order of _mm256_setr_epi8(
//
//	0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
//	0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4).
var popcountAVXLookup = [32]byte{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
}

var popcountAVXLowMask = [32]byte{
	0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
	0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
	0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
	0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
}

func hammingAVX512(a, b []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}

	return hammingAvx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(n))
}
