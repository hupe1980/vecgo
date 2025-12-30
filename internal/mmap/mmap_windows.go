//go:build windows

package mmap

import (
	"os"
	"syscall"
	"unsafe"
)

func mmap(f *os.File, size int) ([]byte, error) {
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, err
	}
	defer syscall.CloseHandle(h)

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		return nil, err
	}

	// Convert uintptr to []byte
	// In Go 1.20+ we could use unsafe.Slice, but let's stick to a compatible way or just unsafe.Slice if we assume modern Go.
	// The user's go.mod likely specifies a version.

	// Using a large array pointer cast is the old way.
	// data := (*[1 << 30]byte)(unsafe.Pointer(addr))[:size:size]

	// Let's use unsafe.Slice which is available in Go 1.17+
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)

	return data, nil
}

func munmap(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	addr := uintptr(unsafe.Pointer(&data[0]))
	return syscall.UnmapViewOfFile(addr)
}

func madvise(data []byte, advice int) error {
	return nil // No-op on Windows
}
