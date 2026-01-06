//go:build amd64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	// Prefer AVX-512 when available, otherwise fall back to AVX2.
	// AVX-512 path needs BW for i8->i32 expansion.
	if cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW {
		squaredL2Int8DequantizedImpl = squaredL2Int8DequantizedAVX512
		buildDistanceTableInt8Impl = buildDistanceTableInt8AVX512
		findNearestCentroidInt8Impl = findNearestCentroidInt8AVX512
		return
	}
	if cpu.X86.HasAVX2 {
		squaredL2Int8DequantizedImpl = squaredL2Int8DequantizedAVX
		buildDistanceTableInt8Impl = buildDistanceTableInt8AVX
		findNearestCentroidInt8Impl = findNearestCentroidInt8AVX
	}
}

func squaredL2Int8DequantizedAVX(query []float32, code []int8, scale, offset float32) float32 {
	var ret float32
	if len(query) > 0 {
		squaredL2Int8DequantizedAvx(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func buildDistanceTableInt8AVX(querySubvec []float32, codebook []int8, subdim int, scale, offset float32, out []float32) {
	if subdim > 0 {
		buildDistanceTableInt8Avx(
			unsafe.Pointer(&querySubvec[0]),
			unsafe.Pointer(&codebook[0]),
			int64(subdim),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&out[0]),
		)
	}
}

func findNearestCentroidInt8AVX(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	var idx int64
	if subdim > 0 {
		findNearestCentroidInt8Avx(
			unsafe.Pointer(&querySubvec[0]),
			unsafe.Pointer(&codebook[0]),
			int64(subdim),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&idx),
		)
	}
	return int(idx)
}

func squaredL2Int8DequantizedAVX512(query []float32, code []int8, scale, offset float32) float32 {
	var ret float32
	if len(query) > 0 {
		squaredL2Int8DequantizedAvx512(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func buildDistanceTableInt8AVX512(querySubvec []float32, codebook []int8, subdim int, scale, offset float32, out []float32) {
	if subdim > 0 {
		buildDistanceTableInt8Avx512(
			unsafe.Pointer(&querySubvec[0]),
			unsafe.Pointer(&codebook[0]),
			int64(subdim),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&out[0]),
		)
	}
}

func findNearestCentroidInt8AVX512(querySubvec []float32, codebook []int8, subdim int, scale, offset float32) int {
	var idx int64
	if subdim > 0 {
		findNearestCentroidInt8Avx512(
			unsafe.Pointer(&querySubvec[0]),
			unsafe.Pointer(&codebook[0]),
			int64(subdim),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&offset),
			unsafe.Pointer(&idx),
		)
	}
	return int(idx)
}
