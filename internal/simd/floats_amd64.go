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
		pqAdcImpl = pqAdcAVX512

		squaredL2BatchImpl = squaredL2BatchAVX512
		dotBatchImpl = dotBatchAVX512
		f16ToF32Impl = f16ToF32AVX512
		sq8L2BatchImpl = sq8L2BatchAVX512
		popcountImpl = popcountAVX512
		hammingImpl = hammingAVX512
	} else if cpu.X86.HasAVX {
		dotImpl = dotAVX
		squaredL2Impl = squaredL2AVX
		pqAdcImpl = pqAdcAVX

		squaredL2BatchImpl = squaredL2BatchAVX
		dotBatchImpl = dotBatchAVX
		f16ToF32Impl = f16ToF32AVX
		sq8L2BatchImpl = sq8L2BatchAVX
		popcountImpl = popcountAVX
		hammingImpl = hammingAVX
	}
}

//go:noescape
func dotProductAvx(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func dotProductAvx512(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func squaredL2Avx(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func squaredL2Avx512(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func pqAdcLookupAvx(table, codes unsafe.Pointer, m int64, result unsafe.Pointer)

//go:noescape
func pqAdcLookupAvx512(table, codes unsafe.Pointer, m int64, result unsafe.Pointer)

//go:noescape
func squaredL2BatchAvx(query, targets unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func squaredL2BatchAvx512(query, targets unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func dotBatchAvx(query, targets unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func dotBatchAvx512(query, targets unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func f16ToF32Avx(in, out unsafe.Pointer, n int64)

//go:noescape
func f16ToF32Avx512(in, out unsafe.Pointer, n int64)

//go:noescape
func sq8L2BatchAvx(query, codes, scales, biases unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func sq8L2BatchAvx512(query, codes, scales, biases unsafe.Pointer, dim, n int64, out unsafe.Pointer)

//go:noescape
func popcountAvx(a unsafe.Pointer, n int) int64

//go:noescape
func popcountAvx512(a unsafe.Pointer, n int) int64

//go:noescape
func hammingAvx(a, b unsafe.Pointer, n int) int64

//go:noescape
func hammingAvx512(a, b unsafe.Pointer, n int) int64

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
		pqAdcLookupAvx(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
	}
	return ret
}

func pqAdcAVX512(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupAvx512(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
	}
	return ret
}

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

func sq8L2BatchAVX512(query []float32, codes []int8, scales []float32, biases []float32, dim int, out []float32) {
	if len(out) > 0 {
		sq8L2BatchAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&codes[0]), unsafe.Pointer(&scales[0]), unsafe.Pointer(&biases[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func popcountAVX(a []byte) int64 {
	if len(a) == 0 {
		return 0
	}
	return popcountAvx(unsafe.Pointer(&a[0]), len(a))
}

func popcountAVX512(a []byte) int64 {
	if len(a) == 0 {
		return 0
	}
	return popcountAvx512(unsafe.Pointer(&a[0]), len(a))
}

func hammingAVX(a, b []byte) int64 {
	if len(a) == 0 {
		return 0
	}
	return hammingAvx(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), len(a))
}

func hammingAVX512(a, b []byte) int64 {
	if len(a) == 0 {
		return 0
	}
	return hammingAvx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), len(a))
}
