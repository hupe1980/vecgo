package simd

// Filter operations for numeric indexing using SIMD-friendly patterns.
// This file contains optimized implementations using loop unrolling
// and memory access patterns that enable compiler auto-vectorization.

// filterRangeF64Impl is the implementation function pointer.
// Uses loop unrolling for better instruction pipelining.
var filterRangeF64Impl = filterRangeF64Unrolled

// FilterRangeF64 returns a bitmask of indices where minVal <= values[i] <= maxVal.
// The result slice will have the same length as the input values slice.
// Each element is 1 if the value is in range, 0 otherwise.
// For performance, the caller should reuse the dst slice when possible.
func FilterRangeF64(values []float64, minVal, maxVal float64, dst []byte) []byte {
	n := len(values)
	if n == 0 {
		return dst[:0]
	}
	// Ensure dst has sufficient capacity
	if cap(dst) < n {
		dst = make([]byte, n)
	} else {
		dst = dst[:n]
	}
	filterRangeF64Impl(values, minVal, maxVal, dst)
	return dst
}

// filterRangeF64Unrolled is an optimized implementation with 8x loop unrolling.
// This pattern is auto-vectorizable by modern compilers and minimizes loop overhead.
func filterRangeF64Unrolled(values []float64, minVal, maxVal float64, dst []byte) {
	n := len(values)
	i := 0

	// Process 8 elements at a time for better pipelining
	for ; i+8 <= n; i += 8 {
		v0, v1, v2, v3 := values[i], values[i+1], values[i+2], values[i+3]
		v4, v5, v6, v7 := values[i+4], values[i+5], values[i+6], values[i+7]

		dst[i] = boolToByte(v0 >= minVal && v0 <= maxVal)
		dst[i+1] = boolToByte(v1 >= minVal && v1 <= maxVal)
		dst[i+2] = boolToByte(v2 >= minVal && v2 <= maxVal)
		dst[i+3] = boolToByte(v3 >= minVal && v3 <= maxVal)
		dst[i+4] = boolToByte(v4 >= minVal && v4 <= maxVal)
		dst[i+5] = boolToByte(v5 >= minVal && v5 <= maxVal)
		dst[i+6] = boolToByte(v6 >= minVal && v6 <= maxVal)
		dst[i+7] = boolToByte(v7 >= minVal && v7 <= maxVal)
	}

	// Handle remainder
	for ; i < n; i++ {
		dst[i] = boolToByte(values[i] >= minVal && values[i] <= maxVal)
	}
}

// boolToByte converts a bool to 0 or 1 without branching.
// The compiler typically optimizes this to a conditional move.
func boolToByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

// filterRangeF64Generic is the simple scalar fallback.
// Currently unused as we have optimized SIMD paths, but kept for reference and testing.
var _ = filterRangeF64Generic

func filterRangeF64Generic(values []float64, minVal, maxVal float64, dst []byte) {
	for i, v := range values {
		if v >= minVal && v <= maxVal {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// filterRangeF64IndicesImpl is the implementation function pointer for direct index extraction.
var filterRangeF64IndicesImpl = filterRangeF64IndicesGeneric

// FilterRangeF64Indices returns the indices of values where minVal <= values[i] <= maxVal.
// This is more efficient than FilterRangeF64 when you need indices directly,
// as it avoids the intermediate bitmask step.
// The dst slice is used to store results. Returns a subslice with matching indices.
func FilterRangeF64Indices(values []float64, minVal, maxVal float64, dst []int32) []int32 {
	n := len(values)
	if n == 0 {
		return dst[:0]
	}
	// Ensure dst has sufficient capacity
	if cap(dst) < n {
		dst = make([]int32, n)
	} else {
		dst = dst[:n]
	}
	count := filterRangeF64IndicesImpl(values, minVal, maxVal, dst)
	return dst[:count]
}

// filterRangeF64IndicesGeneric is the fallback scalar implementation.
func filterRangeF64IndicesGeneric(values []float64, minVal, maxVal float64, dst []int32) int {
	count := 0
	for i, v := range values {
		if v >= minVal && v <= maxVal {
			dst[count] = int32(i)
			count++
		}
	}
	return count
}

// gatherU32Impl is the implementation function pointer.
var gatherU32Impl = gatherU32Generic

// GatherU32 gathers uint32 values from src at the specified indices.
// dst[i] = src[indices[i]] for all i in range.
// This is useful for collecting rowIDs that matched a range filter.
func GatherU32(src []uint32, indices []int32, dst []uint32) []uint32 {
	n := len(indices)
	if n == 0 {
		return dst[:0]
	}
	if cap(dst) < n {
		dst = make([]uint32, n)
	} else {
		dst = dst[:n]
	}
	gatherU32Impl(src, indices, dst)
	return dst
}

// gatherU32Generic is the fallback scalar implementation.
func gatherU32Generic(src []uint32, indices []int32, dst []uint32) {
	for i, idx := range indices {
		dst[i] = src[idx]
	}
}

// CountRangeF64 returns the count of values where minVal <= values[i] <= maxVal.
// This is optimized for cases where you only need the count, not the indices.
var countRangeF64Impl = countRangeF64Generic

func CountRangeF64(values []float64, minVal, maxVal float64) int {
	return countRangeF64Impl(values, minVal, maxVal)
}

func countRangeF64Generic(values []float64, minVal, maxVal float64) int {
	count := 0
	for _, v := range values {
		if v >= minVal && v <= maxVal {
			count++
		}
	}
	return count
}
