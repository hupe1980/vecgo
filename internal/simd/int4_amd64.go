//go:build amd64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	// Prefer AVX-512 when available, otherwise fall back to AVX2.
	// AVX-512 path needs BW for byte manipulation.
	if cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW {
		int4L2DistanceImpl = int4L2DistanceAVX512
		int4L2DistancePrecomputedImpl = int4L2DistancePrecomputedAVX512
		int4L2DistanceBatchImpl = int4L2DistanceBatchAVX512
		return
	}
	if cpu.X86.HasAVX2 {
		int4L2DistanceImpl = int4L2DistanceAVX
		int4L2DistancePrecomputedImpl = int4L2DistancePrecomputedAVX
		int4L2DistanceBatchImpl = int4L2DistanceBatchAVX
	}
}

func int4L2DistanceAVX(query []float32, code []byte, min, diff []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistanceAvx(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&min[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistancePrecomputedAVX(query []float32, code []byte, lookupTable []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistancePrecomputedAvx(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&lookupTable[0]),
			unsafe.Pointer(&ret),
		)
	}
	return ret
}

func int4L2DistanceBatchAVX(query []float32, codes []byte, dim, n int, min, diff []float32, out []float32) {
	if len(query) > 0 && n > 0 {
		int4L2DistanceBatchAvx(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&codes[0]),
			int64(dim),
			int64(n),
			unsafe.Pointer(&min[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&out[0]),
		)
	}
}

func int4L2DistanceAVX512(query []float32, code []byte, min, diff []float32) float32 {
	var ret float32
	if len(query) > 0 {
		int4L2DistanceAvx512(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&code[0]),
			int64(len(query)),
			unsafe.Pointer(&min[0]),
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

func int4L2DistanceBatchAVX512(query []float32, codes []byte, dim, n int, min, diff []float32, out []float32) {
	if len(query) > 0 && n > 0 {
		int4L2DistanceBatchAvx512(
			unsafe.Pointer(&query[0]),
			unsafe.Pointer(&codes[0]),
			int64(dim),
			int64(n),
			unsafe.Pointer(&min[0]),
			unsafe.Pointer(&diff[0]),
			unsafe.Pointer(&out[0]),
		)
	}
}
