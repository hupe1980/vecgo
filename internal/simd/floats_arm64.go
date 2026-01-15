//go:build arm64 && !noasm

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	// Prefer SVE2 when available, otherwise fall back to NEON.
	// SVE2 provides scalable vectors (128-2048 bits) for better performance
	// on modern ARM servers (AWS Graviton 3+, Ampere Altra, etc.)
	if cpu.ARM64.HasSVE2 {
		dotImpl = dotSVE2
		squaredL2Impl = squaredL2SVE2
		scaleImpl = scaleSVE2
		pqAdcImpl = pqAdcSVE2

		squaredL2BatchImpl = squaredL2BatchSVE2
		dotBatchImpl = dotBatchSVE2
		sq8uL2BatchPerDimensionImpl = sq8uL2BatchPerDimensionSVE2
		hammingImpl = hammingSVE2
		return
	}
	if cpu.ARM64.HasASIMD {
		dotImpl = dotNEON
		squaredL2Impl = squaredL2NEON
		scaleImpl = scaleNEON
		pqAdcImpl = pqAdcNEON

		squaredL2BatchImpl = squaredL2BatchNEON
		dotBatchImpl = dotBatchNEON
		sq8uL2BatchPerDimensionImpl = sq8uL2BatchPerDimensionNEON
		hammingImpl = hammingNEON
	}
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

func sq8uL2BatchPerDimensionNEON(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	if len(out) > 0 {
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

func hammingNEON(a, b []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return hammingNeon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(n))
}

// SVE2 implementations

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

func sq8uL2BatchPerDimensionSVE2(query []float32, codes []byte, mins []float32, invScales []float32, dim int, out []float32) {
	if len(out) > 0 {
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

func hammingSVE2(a, b []byte) int64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return hammingSve2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(n))
}
