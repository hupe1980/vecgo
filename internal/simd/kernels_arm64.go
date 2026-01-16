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
	kernelSquaredL2Bounded = squaredL2BoundedNEON
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
	// Bitmap operations - use SIMD for bitwise ops but NOT popcount
	// Go's bits.OnesCount64 compiles to hardware CNT instruction on ARM64,
	// which is faster than our NEON implementation due to less overhead
	kernelAndWords = andWordsNEON
	kernelAndNotWords = andNotWordsNEON
	kernelOrWords = orWordsNEON
	kernelXorWords = xorWordsNEON
	// kernelPopcountWords stays at generic (bits.OnesCount64)
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

func squaredL2BoundedNEON(a, b []float32, bound float32) (float32, bool) {
	if len(a) == 0 {
		return 0, false
	}
	var result float32
	var exceeded int32
	squaredL2BoundedNeon(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		int64(len(a)),
		bound,
		unsafe.Pointer(&result),
		unsafe.Pointer(&exceeded),
	)
	return result, exceeded != 0
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
	kernelSquaredL2Bounded = squaredL2BoundedSVE2
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
	// Bitmap operations - use SIMD for bitwise ops but NOT popcount
	kernelAndWords = andWordsSVE2
	kernelAndNotWords = andNotWordsSVE2
	kernelOrWords = orWordsSVE2
	kernelXorWords = xorWordsSVE2
	// kernelPopcountWords stays at generic (bits.OnesCount64)
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

func squaredL2BoundedSVE2(a, b []float32, bound float32) (float32, bool) {
	if len(a) == 0 {
		return 0, false
	}
	var result float32
	var exceeded int32
	squaredL2BoundedSve2(
		unsafe.Pointer(&a[0]),
		unsafe.Pointer(&b[0]),
		int64(len(a)),
		bound,
		unsafe.Pointer(&result),
		unsafe.Pointer(&exceeded),
	)
	return result, exceeded != 0
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

// ============================================================================
// NEON Bitmap Operations
// ============================================================================

func andWordsNEON(dst, src []uint64) {
	if len(dst) > 0 {
		andWordsNEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

func andNotWordsNEON(dst, src []uint64) {
	if len(dst) > 0 {
		andNotWordsNEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

func orWordsNEON(dst, src []uint64) {
	if len(dst) > 0 {
		orWordsNEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

func xorWordsNEON(dst, src []uint64) {
	if len(dst) > 0 {
		xorWordsNEONAsm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

// NOTE: popcountWordsNEON removed - Go's bits.OnesCount64 compiles to hardware
// CNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.

// ============================================================================
// SVE2 Bitmap Operations
// ============================================================================

func andWordsSVE2(dst, src []uint64) {
	if len(dst) > 0 {
		andWordsSVE2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

func andNotWordsSVE2(dst, src []uint64) {
	if len(dst) > 0 {
		andNotWordsSVE2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

func orWordsSVE2(dst, src []uint64) {
	if len(dst) > 0 {
		orWordsSVE2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

func xorWordsSVE2(dst, src []uint64) {
	if len(dst) > 0 {
		xorWordsSVE2Asm(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), int64(len(dst)))
	}
}

// NOTE: popcountWordsSVE2 removed - Go's bits.OnesCount64 compiles to hardware
// CNT instruction and is faster than explicit SIMD due to reduced overhead.
// See: CRoaring and bits-and-blooms/bitset implementations.
