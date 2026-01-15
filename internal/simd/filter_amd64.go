//go:build amd64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	// Filter operations - prefer AVX-512 when available
	if cpu.X86.HasAVX512F {
		filterRangeF64Impl = filterRangeF64AVX512Wrapper
		filterRangeF64IndicesImpl = filterRangeF64IndicesAVX512Wrapper
		countRangeF64Impl = countRangeF64AVX512Wrapper
		gatherU32Impl = gatherU32AVX512Wrapper
		return
	}
	if cpu.X86.HasAVX {
		filterRangeF64Impl = filterRangeF64AVXWrapper
		filterRangeF64IndicesImpl = filterRangeF64IndicesAVXWrapper
		countRangeF64Impl = countRangeF64AVXWrapper
		gatherU32Impl = gatherU32AVXWrapper
	}
}

// AVX wrappers

func filterRangeF64AVXWrapper(values []float64, minVal, maxVal float64, dst []byte) {
	if len(values) > 0 {
		filterRangeF64Avx(
			unsafe.Pointer(&values[0]),
			int64(len(values)),
			minVal,
			maxVal,
			unsafe.Pointer(&dst[0]),
		)
	}
}

func filterRangeF64IndicesAVXWrapper(values []float64, minVal, maxVal float64, dst []int32) int {
	if len(values) == 0 {
		return 0
	}
	return int(filterRangeF64IndicesAvx(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&dst[0]),
	))
}

func countRangeF64AVXWrapper(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	return int(countRangeF64Avx(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
	))
}

func gatherU32AVXWrapper(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Avx(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}

// AVX-512 wrappers

func filterRangeF64AVX512Wrapper(values []float64, minVal, maxVal float64, dst []byte) {
	if len(values) > 0 {
		filterRangeF64Avx512(
			unsafe.Pointer(&values[0]),
			int64(len(values)),
			minVal,
			maxVal,
			unsafe.Pointer(&dst[0]),
		)
	}
}

func filterRangeF64IndicesAVX512Wrapper(values []float64, minVal, maxVal float64, dst []int32) int {
	if len(values) == 0 {
		return 0
	}
	return int(filterRangeF64IndicesAvx512(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&dst[0]),
	))
}

func countRangeF64AVX512Wrapper(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	return int(countRangeF64Avx512(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
	))
}

func gatherU32AVX512Wrapper(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Avx512(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}
