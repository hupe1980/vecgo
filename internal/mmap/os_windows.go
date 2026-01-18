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

func osMapAnon(size int) ([]byte, func([]byte) error, error) {
	// Use VirtualAlloc with MEM_RESERVE | MEM_COMMIT for anonymous memory.
	// Unlike CreateFileMapping (which requires paging file commitment upfront),
	// VirtualAlloc with MEM_COMMIT uses demand-paging: pages are only backed
	// by physical memory when first accessed, similar to Unix mmap behavior.
	//
	// This avoids "paging file is too small" errors on systems with limited
	// paging file space (e.g., CI runners).
	addr, err := windows.VirtualAlloc(0, uintptr(size),
		windows.MEM_RESERVE|windows.MEM_COMMIT, windows.PAGE_READWRITE)
	if err != nil {
		return nil, nil, err
	}

	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)

	return data, func(b []byte) error {
		// VirtualFree with MEM_RELEASE frees the entire region
		return windows.VirtualFree(addr, 0, windows.MEM_RELEASE)
	}, nil
}

func osAdvise(data []byte, pattern AccessPattern) error {
	// Windows does not have a direct equivalent to madvise.
	// PrefetchVirtualMemory could be used for AccessWillNeed, but requires
	// Windows 8+ and more complex setup. For now, this is a no-op.
	// The OS page cache will still work effectively for sequential access.
	_ = data
	_ = pattern
	return nil
}
