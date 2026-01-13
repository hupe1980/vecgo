//go:build arm64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	// Prefer SVE2 when available, otherwise fall back to NEON.
	if cpu.ARM64.HasSVE2 {
		int4L2DistanceImpl = int4L2DistanceSVE2
		int4L2DistancePrecomputedImpl = int4L2DistancePrecomputedSVE2
		int4L2DistanceBatchImpl = int4L2DistanceBatchSVE2
		return
	}
	// NEON is mandatory on arm64.
	int4L2DistanceImpl = int4L2DistanceNEON
	int4L2DistancePrecomputedImpl = int4L2DistancePrecomputedNEON
	int4L2DistanceBatchImpl = int4L2DistanceBatchNEON
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

// SVE2 implementations

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
