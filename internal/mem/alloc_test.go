package mem

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestAllocAligned(t *testing.T) {
	sizes := []int{1, 10, 63, 64, 65, 100, 1024}

	for _, size := range sizes {
		buf := AllocAligned(size)
		assert.Len(t, buf, size)

		ptr := unsafe.Pointer(&buf[0])
		addr := uintptr(ptr)
		assert.Equal(t, uintptr(0), addr%Alignment, "Address %d should be aligned to %d for size %d", addr, Alignment, size)
	}

	assert.Nil(t, AllocAligned(0))
}

func TestAllocAlignedFloat32(t *testing.T) {
	sizes := []int{1, 10, 16, 17, 100, 1024}

	for _, size := range sizes {
		buf := AllocAlignedFloat32(size)
		assert.Len(t, buf, size)

		ptr := unsafe.Pointer(&buf[0])
		addr := uintptr(ptr)
		assert.Equal(t, uintptr(0), addr%Alignment, "Address %d should be aligned to %d for size %d", addr, Alignment, size)
	}

	assert.Nil(t, AllocAlignedFloat32(0))
}
