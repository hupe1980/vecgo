//go:build !windows

package mmap

import (
	"os"

	"golang.org/x/sys/unix"
)

func mmap(f *os.File, size int) ([]byte, error) {
	return unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
}

func munmap(data []byte) error {
	return unix.Munmap(data)
}
