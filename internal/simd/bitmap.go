package simd

import "math/bits"

// ==============================================================================
// Bitmap SIMD Operations
// ==============================================================================
//
// These operations are optimized for QueryBitmap in the bitmap package.
// They operate on []uint64 representing bit arrays, with SIMD acceleration
// for AND, OR, XOR, ANDNOT, and POPCOUNT operations.

// Kernel function pointers for bitmap operations.
// Generic implementations are the default; platform-specific init()
// functions override with SIMD versions when available.
var (
	kernelAndWords      = andWordsGeneric
	kernelAndNotWords   = andNotWordsGeneric
	kernelOrWords       = orWordsGeneric
	kernelXorWords      = xorWordsGeneric
	kernelPopcountWords = popcountWordsGeneric
)

// AndWords performs dst[i] &= src[i] for all words.
// SIMD-accelerated on supported platforms.
func AndWords(dst, src []uint64) {
	kernelAndWords(dst, src)
}

// AndNotWords performs dst[i] &= ^src[i] for all words.
// SIMD-accelerated on supported platforms.
func AndNotWords(dst, src []uint64) {
	kernelAndNotWords(dst, src)
}

// OrWords performs dst[i] |= src[i] for all words.
// SIMD-accelerated on supported platforms.
func OrWords(dst, src []uint64) {
	kernelOrWords(dst, src)
}

// XorWords performs dst[i] ^= src[i] for all words.
// SIMD-accelerated on supported platforms.
func XorWords(dst, src []uint64) {
	kernelXorWords(dst, src)
}

// PopcountWords counts all set bits across words.
// SIMD-accelerated on supported platforms using POPCNT/CNT instructions.
func PopcountWords(words []uint64) int {
	return kernelPopcountWords(words)
}

// ==============================================================================
// Generic implementations
// ==============================================================================

func andWordsGeneric(dst, src []uint64) {
	// Process 4 words at a time (unrolled)
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		dst[i] &= src[i]
		dst[i+1] &= src[i+1]
		dst[i+2] &= src[i+2]
		dst[i+3] &= src[i+3]
	}
	for ; i < len(dst); i++ {
		dst[i] &= src[i]
	}
}

func andNotWordsGeneric(dst, src []uint64) {
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		dst[i] &= ^src[i]
		dst[i+1] &= ^src[i+1]
		dst[i+2] &= ^src[i+2]
		dst[i+3] &= ^src[i+3]
	}
	for ; i < len(dst); i++ {
		dst[i] &= ^src[i]
	}
}

func orWordsGeneric(dst, src []uint64) {
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		dst[i] |= src[i]
		dst[i+1] |= src[i+1]
		dst[i+2] |= src[i+2]
		dst[i+3] |= src[i+3]
	}
	for ; i < len(dst); i++ {
		dst[i] |= src[i]
	}
}

func xorWordsGeneric(dst, src []uint64) {
	i := 0
	for ; i+4 <= len(dst); i += 4 {
		dst[i] ^= src[i]
		dst[i+1] ^= src[i+1]
		dst[i+2] ^= src[i+2]
		dst[i+3] ^= src[i+3]
	}
	for ; i < len(dst); i++ {
		dst[i] ^= src[i]
	}
}

func popcountWordsGeneric(words []uint64) int {
	count := 0
	// Process 4 words at a time
	i := 0
	for ; i+4 <= len(words); i += 4 {
		count += bits.OnesCount64(words[i])
		count += bits.OnesCount64(words[i+1])
		count += bits.OnesCount64(words[i+2])
		count += bits.OnesCount64(words[i+3])
	}
	for ; i < len(words); i++ {
		count += bits.OnesCount64(words[i])
	}
	return count
}
