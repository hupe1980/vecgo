//go:build amd64 && !noasm

package simd

import "unsafe"

// init sets the SIMD kernel pointers based on the active ISA.
// This runs after capability_amd64.go init() has detected CPU features
// and selected the active ISA.
func init() {
	switch activeISA {
	case AVX2:
		setAVX2Kernels()
	case AVX512:
		setAVX512Kernels()
	}
}

// ============================================================================
// AVX2 Kernels
// ============================================================================

// Precomputed offset tables for PQ ADC lookup (256 entries per subvector)
var pqAdcOffsets = func() []int32 {
	offsets := make([]int32, 256)
	for i := range offsets {
		offsets[i] = int32(i * 256 * 4) // 256 floats per subvector, 4 bytes per float
	}
	return offsets
}()

func setAVX2Kernels() {
	kernelDot = dotAVX2
	kernelSquaredL2 = squaredL2AVX2
	kernelSquaredL2Bounded = squaredL2BoundedAVX2
	kernelScale = scaleAVX2
	kernelPqAdc = pqAdcAVX2
	kernelDotBatch = dotBatchAVX2
	kernelSquaredL2Batch = squaredL2BatchAVX2
	kernelHamming = hammingAVX2
	kernelSQ8uL2BatchPerDim = sq8uL2BatchPerDimensionAVX2
	kernelInt4L2Distance = int4L2DistanceAVX2
	kernelInt4L2DistancePrecomputed = int4L2DistancePrecomputedAVX2
	kernelInt4L2DistanceBatch = int4L2DistanceBatchAVX2
	kernelFilterRangeF64 = filterRangeF64AVX2
	kernelFilterRangeF64Indices = filterRangeF64IndicesAVX2
	kernelCountRangeF64 = countRangeF64AVX2
	kernelGatherU32 = gatherU32AVX2
}

func dotAVX2(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		dotProductAvx2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2AVX2(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		squaredL2Avx2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2BoundedAVX2(a, b []float32, bound float32) (float32, bool) {
	if len(a) == 0 {
		return 0, false
	}
	var result float32
	var exceeded int32
	squaredL2BoundedAvx2(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		int64(len(a)),
		bound,
		unsafe.Pointer(&result),
		unsafe.Pointer(&exceeded),
	)
	return result, exceeded != 0
}

func scaleAVX2(a []float32, scalar float32) {
	if len(a) > 0 {
		scaleAvx2(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&scalar))
	}
}

func pqAdcAVX2(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupAvx2(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&codes[0]),
			int64(m),
			unsafe.Pointer(&ret),
			unsafe.Pointer(&pqAdcOffsets[0]),
		)
	}
	return ret
}

func squaredL2BatchAVX2(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		squaredL2BatchAvx2(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func dotBatchAVX2(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		dotBatchAvx2(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func hammingAVX2(a, b []byte) int {
	var ret int64
	if len(a) > 0 {
		hammingAvx2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return int(ret)
}

func sq8uL2BatchPerDimensionAVX2(query []float32, codes []byte, mins, invScales []float32, dim int, out []float32) {
	if len(out) > 0 && dim > 0 {
		sq8uL2BatchPerDimensionAvx2(
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

func int4L2DistanceAVX2(query []float32, code []byte, minVal, diff []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistanceAvx2(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&minVal[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistancePrecomputedAVX2(query []float32, code []byte, lookupTable []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistancePrecomputedAvx2(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&lookupTable[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistanceBatchAVX2(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	if len(query) > 0 && n > 0 {
		int4L2DistanceBatchAvx2(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&codes[0]),
			int64(dim),
			int64(n),
			unsafe.Pointer(&minVal[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&out[0]),
		)
	}
}

func filterRangeF64AVX2(values []float64, minVal, maxVal float64, dst []byte) {
	if len(values) > 0 {
		filterRangeF64Avx2(
			unsafe.Pointer(&values[0]),
			int64(len(values)),
			minVal,
			maxVal,
			unsafe.Pointer(&dst[0]),
		)
	}
}

func filterRangeF64IndicesAVX2(values []float64, minVal, maxVal float64, indices []int32) int {
	if len(values) == 0 {
		return 0
	}
	count := filterRangeF64IndicesAvx2(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&indices[0]),
	)
	return int(count)
}

func countRangeF64AVX2(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	count := countRangeF64Avx2(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
	)
	return int(count)
}

func gatherU32AVX2(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Avx2(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}

// ============================================================================
// AVX-512 Kernels
// ============================================================================

func setAVX512Kernels() {
	kernelDot = dotAVX512
	kernelSquaredL2 = squaredL2AVX512
	kernelSquaredL2Bounded = squaredL2BoundedAVX512
	kernelScale = scaleAVX512
	kernelPqAdc = pqAdcAVX512
	kernelDotBatch = dotBatchAVX512
	kernelSquaredL2Batch = squaredL2BatchAVX512
	kernelHamming = hammingAVX512
	kernelSQ8uL2BatchPerDim = sq8uL2BatchPerDimensionAVX512
	kernelInt4L2Distance = int4L2DistanceAVX512
	kernelInt4L2DistancePrecomputed = int4L2DistancePrecomputedAVX512
	kernelInt4L2DistanceBatch = int4L2DistanceBatchAVX512
	kernelFilterRangeF64 = filterRangeF64AVX512
	kernelFilterRangeF64Indices = filterRangeF64IndicesAVX512
	kernelCountRangeF64 = countRangeF64AVX512
	kernelGatherU32 = gatherU32AVX512
}

func dotAVX512(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		dotProductAvx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
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

func squaredL2BoundedAVX512(a, b []float32, bound float32) (float32, bool) {
	if len(a) == 0 {
		return 0, false
	}
	var result float32
	var exceeded int32
	squaredL2BoundedAvx512(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		int64(len(a)),
		bound,
		unsafe.Pointer(&result),
		unsafe.Pointer(&exceeded),
	)
	return result, exceeded != 0
}

func scaleAVX512(a []float32, scalar float32) {
	if len(a) > 0 {
		scaleAvx512(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&scalar))
	}
}

func pqAdcAVX512(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupAvx512(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&codes[0]),
			int64(m),
			unsafe.Pointer(&ret),
			unsafe.Pointer(&pqAdcOffsets[0]),
		)
	}
	return ret
}

func squaredL2BatchAVX512(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		squaredL2BatchAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func dotBatchAVX512(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		dotBatchAvx512(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func hammingAVX512(a, b []byte) int {
	var ret int64
	if len(a) > 0 {
		hammingAvx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return int(ret)
}

func sq8uL2BatchPerDimensionAVX512(query []float32, codes []byte, mins, invScales []float32, dim int, out []float32) {
	if len(out) > 0 && dim > 0 {
		sq8uL2BatchPerDimensionAvx512(
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

func int4L2DistanceAVX512(query []float32, code []byte, minVal, diff []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistanceAvx512(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&minVal[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistancePrecomputedAVX512(query []float32, code []byte, lookupTable []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistancePrecomputedAvx512(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&lookupTable[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistanceBatchAVX512(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	if len(query) > 0 && n > 0 {
		int4L2DistanceBatchAvx512(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&codes[0]),
			int64(dim),
			int64(n),
			unsafe.Pointer(&minVal[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&out[0]),
		)
	}
}

func filterRangeF64AVX512(values []float64, minVal, maxVal float64, dst []byte) {
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

func filterRangeF64IndicesAVX512(values []float64, minVal, maxVal float64, indices []int32) int {
	if len(values) == 0 {
		return 0
	}
	count := filterRangeF64IndicesAvx512(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&indices[0]),
	)
	return int(count)
}

func countRangeF64AVX512(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	count := countRangeF64Avx512(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
	)
	return int(count)
}

func gatherU32AVX512(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Avx512(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}
