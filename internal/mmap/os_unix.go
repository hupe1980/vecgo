//go:build unix || linux || darwin || freebsd || openbsd || netbsd

package mmap

import (
	"os"

	"golang.org/x/sys/unix"
)

func osMap(f *os.File, size int) ([]byte, func([]byte) error, error) {
	prot := unix.PROT_READ
	flags := unix.MAP_SHARED

	// Use generic Mmap.
	// Note: Fd() returns uintptr, Mmap expects int on some platforms, but unix package handles it.
	data, err := unix.Mmap(int(f.Fd()), 0, size, prot, flags)
	if err != nil {
		return nil, nil, err
	}

	return data, unix.Munmap, nil
}

func osMapAnon(size int) ([]byte, func([]byte) error, error) {
	prot := unix.PROT_READ | unix.PROT_WRITE
	flags := unix.MAP_ANON | unix.MAP_PRIVATE

	data, err := unix.Mmap(-1, 0, size, prot, flags)
	if err != nil {
		return nil, nil, err
	}

	return data, unix.Munmap, nil
}

func osAdvise(data []byte, pattern AccessPattern) error {
	if len(data) == 0 {
		return nil
	}

	var advice int
	switch pattern {
	case AccessSequential:
		advice = unix.MADV_SEQUENTIAL
	case AccessRandom:
		advice = unix.MADV_RANDOM
	case AccessWillNeed:
		advice = unix.MADV_WILLNEED
	case AccessDontNeed:
		advice = unix.MADV_DONTNEED
	default:
		advice = unix.MADV_NORMAL
	}

	// On Linux, madvise requires page-aligned addresses.
	// If the slice isn't page-aligned, we silently succeed since
	// the hint is advisory and non-critical.
	err := unix.Madvise(data, advice)
	if err == unix.EINVAL {
		// Likely a page alignment issue on Linux - ignore it
		return nil
	}
	return err
}
