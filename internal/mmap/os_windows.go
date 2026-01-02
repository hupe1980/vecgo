//go:build windows

package mmap

import (
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

func osMap(f *os.File, size int) ([]byte, func([]byte) error, error) {
	if size == 0 {
		return nil, nil, nil
	}

	// Create file mapping object
	// PAGE_READONLY for read-only access
	h, err := windows.CreateFileMapping(windows.Handle(f.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, nil, err
	}
	// We can close the handle immediately after creating the view, as the view holds a reference.
	defer windows.CloseHandle(h)

	// Map view of file
	// FILE_MAP_READ for read access
	addr, err := windows.MapViewOfFile(h, windows.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		return nil, nil, err
	}

	// Convert uintptr to []byte
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)

	return data, func(b []byte) error {
		// We need the address to unmap.
		// We capture 'addr' in the closure which is safer than reconstructing from slice.
		return windows.UnmapViewOfFile(addr)
	}, nil
}

func osAdvise(data []byte, pattern AccessPattern) error {
	// Windows does not have a direct equivalent to madvise for file mappings in the same way.
	// PrefetchVirtualMemory could be used, but it's more complex.
	// For now, this is a no-op as per the plan.
	return nil
}
