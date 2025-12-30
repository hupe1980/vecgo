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

func madvise(data []byte, advice int) error {
	var sysAdvice int
	switch advice {
	case AdviceNormal:
		sysAdvice = unix.MADV_NORMAL
	case AdviceRandom:
		sysAdvice = unix.MADV_RANDOM
	case AdviceSequential:
		sysAdvice = unix.MADV_SEQUENTIAL
	case AdviceWillNeed:
		sysAdvice = unix.MADV_WILLNEED
	case AdviceDontNeed:
		sysAdvice = unix.MADV_DONTNEED
	default:
		return nil
	}
	return unix.Madvise(data, sysAdvice)
}
