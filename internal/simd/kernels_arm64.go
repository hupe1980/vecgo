//go:build arm64 && !noasm

package simd

import "unsafe"

// init sets the SIMD kernel pointers based on the active ISA.
// This runs after capability_arm64.go init() has detected CPU features
// and selected the active ISA.
func init() {
	switch activeISA {
	case NEON:
		setNEONKernels()
	case SVE2:
		setSVE2Kernels()
	}
}

// ============================================================================
// NEON Kernels
// ============================================================================

func setNEONKernels() {
	kernelDot = dotNEON
	kernelSquaredL2 = squaredL2NEON
	kernelScale = scaleNEON
	kernelPqAdc = pqAdcNEON
	kernelDotBatch = dotBatchNEON
	kernelSquaredL2Batch = squaredL2BatchNEON
	kernelHamming = hammingNEON
	kernelSQ8uL2BatchPerDim = sq8uL2BatchPerDimensionNEON
	kernelInt4L2Distance = int4L2DistanceNEON
	kernelInt4L2DistancePrecomputed = int4L2DistancePrecomputedNEON
	kernelInt4L2DistanceBatch = int4L2DistanceBatchNEON
	kernelFilterRangeF64 = filterRangeF64NEON
	kernelFilterRangeF64Indices = filterRangeF64IndicesNEON
	kernelCountRangeF64 = countRangeF64NEON
	kernelGatherU32 = gatherU32NEON
}

func dotNEON(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		dotProductNeon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2NEON(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		squaredL2Neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
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

func hammingNEON(a, b []byte) int {
	if len(a) == 0 {
		return 0
	}
	return int(hammingNeon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a))))
}

func sq8uL2BatchPerDimensionNEON(query []float32, codes []byte, mins, invScales []float32, dim int, out []float32) {
	if len(out) > 0 && dim > 0 {
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

func int4L2DistanceNEON(query []float32, code []byte, minVal, diff []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistanceNeon(
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

func int4L2DistancePrecomputedNEON(query []float32, code []byte, lookupTable []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistancePrecomputedNeon(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&lookupTable[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistanceBatchNEON(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	if len(query) > 0 && n > 0 {
		int4L2DistanceBatchNeon(
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

func filterRangeF64NEON(values []float64, minVal, maxVal float64, dst []byte) {
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

func filterRangeF64IndicesNEON(values []float64, minVal, maxVal float64, indices []int32) int {
	if len(values) == 0 {
		return 0
	}
	var count int64
	filterRangeF64IndicesNeon(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&indices[0]),
		unsafe.Pointer(&count),
	)
	return int(count)
}

func countRangeF64NEON(values []float64, minVal, maxVal float64) int {
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

func gatherU32NEON(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Neon(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}

// ============================================================================
// SVE2 Kernels
// ============================================================================

func setSVE2Kernels() {
	kernelDot = dotSVE2
	kernelSquaredL2 = squaredL2SVE2
	kernelScale = scaleSVE2
	kernelPqAdc = pqAdcSVE2
	kernelDotBatch = dotBatchSVE2
	kernelSquaredL2Batch = squaredL2BatchSVE2
	kernelHamming = hammingSVE2
	kernelSQ8uL2BatchPerDim = sq8uL2BatchPerDimensionSVE2
	kernelInt4L2Distance = int4L2DistanceSVE2
	kernelInt4L2DistancePrecomputed = int4L2DistancePrecomputedSVE2
	kernelInt4L2DistanceBatch = int4L2DistanceBatchSVE2
	kernelFilterRangeF64 = filterRangeF64SVE2
	kernelFilterRangeF64Indices = filterRangeF64IndicesSVE2
	kernelCountRangeF64 = countRangeF64SVE2
	kernelGatherU32 = gatherU32SVE2
}

func dotSVE2(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		dotProductSve2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2SVE2(a, b []float32) float32 {
	var ret float32
	if len(a) > 0 {
		squaredL2Sve2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}
	return ret
}

func scaleSVE2(a []float32, scalar float32) {
	if len(a) > 0 {
		scaleSve2(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&scalar))
	}
}

func pqAdcSVE2(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		pqAdcLookupSve2(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
	}
	return ret
}

func squaredL2BatchSVE2(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		squaredL2BatchSve2(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func dotBatchSVE2(query []float32, targets []float32, dim int, out []float32) {
	if len(out) > 0 {
		dotBatchSve2(unsafe.Pointer(&query[0]), unsafe.Pointer(&targets[0]), int64(dim), int64(len(out)), unsafe.Pointer(&out[0]))
	}
}

func hammingSVE2(a, b []byte) int {
	if len(a) == 0 {
		return 0
	}
	return int(hammingSve2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a))))
}

func sq8uL2BatchPerDimensionSVE2(query []float32, codes []byte, mins, invScales []float32, dim int, out []float32) {
	if len(out) > 0 && dim > 0 {
		sq8uL2BatchPerDimensionSve2(
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

func int4L2DistanceSVE2(query []float32, code []byte, minVal, diff []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistanceSve2(
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

func int4L2DistancePrecomputedSVE2(query []float32, code []byte, lookupTable []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistancePrecomputedSve2(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&lookupTable[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistanceBatchSVE2(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	if len(query) > 0 && n > 0 {
		int4L2DistanceBatchSve2(
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

func filterRangeF64SVE2(values []float64, minVal, maxVal float64, dst []byte) {
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

func filterRangeF64IndicesSVE2(values []float64, minVal, maxVal float64, indices []int32) int {
	if len(values) == 0 {
		return 0
	}
	var count int64
	filterRangeF64IndicesSve2(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&indices[0]),
		unsafe.Pointer(&count),
	)
	return int(count)
}

func countRangeF64SVE2(values []float64, minVal, maxVal float64) int {
	if len(values) == 0 {
		return 0
	}
	var count int64
	countRangeF64Sve2(
		unsafe.Pointer(&values[0]),
		int64(len(values)),
		minVal,
		maxVal,
		unsafe.Pointer(&count),
	)
	return int(count)
}

func gatherU32SVE2(src []uint32, indices []int32, dst []uint32) {
	if len(indices) > 0 {
		gatherU32Sve2(
			unsafe.Pointer(&src[0]),
			unsafe.Pointer(&indices[0]),
			int64(len(indices)),
			unsafe.Pointer(&dst[0]),
		)
	}
}
