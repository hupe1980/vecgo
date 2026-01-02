// Package mem provides memory allocation utilities.
package mem

import (
	"unsafe"
)

// Alignment is the byte alignment required for AVX-512 (64 bytes).
const Alignment = 64

// AllocAligned allocates a byte slice of the given size with 64-byte alignment.
// The returned slice is guaranteed to start at a memory address divisible by 64.
//
// Note: This function allocates slightly more memory than requested to ensure alignment.
// The underlying array is kept alive by the returned slice.
func AllocAligned(size int) []byte {
	if size == 0 {
		return nil
	}

	// Allocate size + alignment to ensure we can find an aligned offset
	// We need enough space to shift the start pointer up to Alignment-1 bytes
	totalSize := size + Alignment
	buf := make([]byte, totalSize)

	// Calculate the offset to the first aligned byte
	ptr := unsafe.Pointer(&buf[0]) //nolint:gosec // unsafe is required for memory alignment
	addr := uintptr(ptr)
	offset := (Alignment - (addr & (Alignment - 1))) & (Alignment - 1)

	// Return the slice starting at the aligned offset
	return buf[offset : offset+uintptr(size)]
}

// AllocAlignedFloat32 allocates a float32 slice of the given size with 64-byte alignment.
// The returned slice is guaranteed to start at a memory address divisible by 64.
func AllocAlignedFloat32(size int) []float32 {
	if size == 0 {
		return nil
	}

	byteSize := size * 4
	byteSlice := AllocAligned(byteSize)

	// Convert []byte to []float32
	// This is safe because AllocAligned guarantees 64-byte alignment,
	// which is also 4-byte aligned (required for float32).
	ptr := unsafe.Pointer(&byteSlice[0])       //nolint:gosec // unsafe is required for memory alignment
	return unsafe.Slice((*float32)(ptr), size) //nolint:gosec // unsafe is required for memory alignment
}

// AllocAlignedInt8 allocates an int8 slice of the given size with 64-byte alignment.
func AllocAlignedInt8(size int) []int8 {
	if size == 0 {
		return nil
	}
	byteSlice := AllocAligned(size)
	ptr := unsafe.Pointer(&byteSlice[0])    //nolint:gosec // unsafe is required for memory alignment
	return unsafe.Slice((*int8)(ptr), size) //nolint:gosec // unsafe is required for memory alignment
}
