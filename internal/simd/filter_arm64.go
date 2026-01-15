//go:build arm64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	// Filter operations - prefer SVE2 when available, otherwise fall back to NEON
	if cpu.ARM64.HasSVE2 {
		filterRangeF64Impl = filterRangeF64SVE2Wrapper
		filterRangeF64IndicesImpl = filterRangeF64IndicesSVE2Wrapper
		countRangeF64Impl = countRangeF64SVE2Wrapper
		gatherU32Impl = gatherU32SVE2Wrapper
		return
	}
	if cpu.ARM64.HasASIMD {
		filterRangeF64Impl = filterRangeF64NEONWrapper
		filterRangeF64IndicesImpl = filterRangeF64IndicesNEONWrapper
		countRangeF64Impl = countRangeF64NEONWrapper
		gatherU32Impl = gatherU32NEONWrapper
	}
}

// SVE2 wrappers

func filterRangeF64SVE2Wrapper(values []float64, minVal, maxVal float64, dst []byte) {
	if len(values) > 0 {
		filterRangeF64Sve2(
			unsafe.Pointer(&values[0]),
			int64(len(values)),
			minVal,
			maxVal,
			unsafe.Pointer(&dst[0]),
		)
	}
}

func filterRangeF64IndicesSVE2Wrapper(values []float64, minVal, maxVal float64, dst []int32) int {
	if len(values) == 0 {
		return 0
	}
	return int(filterRangeF64IndicesSve2(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&dst[0]),
	))
}

func countRangeF64SVE2Wrapper(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	return int(countRangeF64Sve2(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
	))
}

func gatherU32SVE2Wrapper(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Sve2(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}

// NEON wrappers

func filterRangeF64NEONWrapper(values []float64, minVal, maxVal float64, dst []byte) {
	if len(values) > 0 {
		filterRangeF64Neon(
			unsafe.Pointer(&values[0]),
			int64(len(values)),
			minVal,
			maxVal,
			unsafe.Pointer(&dst[0]),
		)
	}
}

func filterRangeF64IndicesNEONWrapper(values []float64, minVal, maxVal float64, dst []int32) int {
	if len(values) == 0 {
		return 0
	}
	var count int64
	filterRangeF64IndicesNeon(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(&count),
	)
	return int(count)
}

func countRangeF64NEONWrapper(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	var count int64
	countRangeF64Neon(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&count),
	)
	return int(count)
}

func gatherU32NEONWrapper(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Neon(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}
