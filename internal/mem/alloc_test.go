package mem

import (
	"fmt"
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
	assert.Nil(t, AllocAligned(-1))
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
	assert.Nil(t, AllocAlignedFloat32(-1))
}

func TestAllocAlignedInt8(t *testing.T) {
	sizes := []int{1, 10, 63, 64, 65, 100, 1024}

	for _, size := range sizes {
		buf := AllocAlignedInt8(size)
		assert.Len(t, buf, size)

		ptr := unsafe.Pointer(&buf[0])
		addr := uintptr(ptr)
		assert.Equal(t, uintptr(0), addr%Alignment, "Address %d should be aligned to %d for size %d", addr, Alignment, size)
	}

	assert.Nil(t, AllocAlignedInt8(0))
	assert.Nil(t, AllocAlignedInt8(-1))
}

func BenchmarkAllocAligned(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = AllocAligned(size)
			}
		})
	}
}

func BenchmarkAllocAlignedFloat32(b *testing.B) {
	sizes := []int{16, 64, 256, 1024}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = AllocAlignedFloat32(size)
			}
		})
	}
}
