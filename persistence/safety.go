// Package persistence provides verified unsafe operations with runtime safety checks.
package persistence

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

var (
	// ErrUnsupportedArchitecture is returned when running on unsupported CPU architecture
	ErrUnsupportedArchitecture = errors.New("unsupported architecture: only amd64 and arm64 are supported")

	// ErrBigEndian is returned when running on big-endian systems
	ErrBigEndian = errors.New("big-endian systems are not supported")

	// ErrUnalignedAccess is returned when attempting unaligned memory access
	ErrUnalignedAccess = errors.New("unaligned memory access detected")
)

// init performs startup validation of platform requirements
func init() {
	if err := validatePlatform(); err != nil {
		panic(fmt.Sprintf("vecgo/persistence: %v", err))
	}
}

// validatePlatform checks if the current platform supports unsafe operations
func validatePlatform() error {
	// Check architecture
	arch := runtime.GOARCH
	if arch != "amd64" && arch != "arm64" {
		return fmt.Errorf("%w: %s", ErrUnsupportedArchitecture, arch)
	}

	// Check endianness (must be little-endian)
	if !isLittleEndian() {
		return ErrBigEndian
	}

	return nil
}

// isLittleEndian checks if the system is little-endian
func isLittleEndian() bool {
	var test uint16 = 0x0001
	firstByte := *(*byte)(unsafe.Pointer(&test))
	return firstByte == 1
}

// validateFloat32SliceAlignment checks if a float32 slice is properly aligned
func validateFloat32SliceAlignment(vec []float32) error {
	if len(vec) == 0 {
		return nil
	}

	ptr := uintptr(unsafe.Pointer(&vec[0]))
	if ptr%4 != 0 {
		return fmt.Errorf("%w: float32 slice at address 0x%x", ErrUnalignedAccess, ptr)
	}

	return nil
}

// validateUint32SliceAlignment checks if a uint32 slice is properly aligned
func validateUint32SliceAlignment(slice []uint32) error {
	if len(slice) == 0 {
		return nil
	}

	ptr := uintptr(unsafe.Pointer(&slice[0]))
	if ptr%4 != 0 {
		return fmt.Errorf("%w: uint32 slice at address 0x%x", ErrUnalignedAccess, ptr)
	}

	return nil
}

// validateUint64SliceAlignment checks if a uint64 slice is properly aligned
func validateUint64SliceAlignment(slice []uint64) error {
	if len(slice) == 0 {
		return nil
	}

	ptr := uintptr(unsafe.Pointer(&slice[0]))
	if ptr%8 != 0 {
		return fmt.Errorf("%w: uint64 slice at address 0x%x", ErrUnalignedAccess, ptr)
	}

	return nil
}

// PlatformInfo returns information about the current platform
func PlatformInfo() string {
	endian := "little-endian"
	if !isLittleEndian() {
		endian = "big-endian"
	}
	return fmt.Sprintf("GOOS=%s GOARCH=%s endianness=%s", runtime.GOOS, runtime.GOARCH, endian)
}
