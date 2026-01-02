package mmap

import "errors"

// AccessPattern provides hints to the kernel about how the data will be accessed.
type AccessPattern int

const (
	// AccessDefault is the default access pattern (no specific advice).
	AccessDefault AccessPattern = iota
	// AccessSequential expects data to be accessed sequentially.
	AccessSequential
	// AccessRandom expects data to be accessed randomly.
	AccessRandom
	// AccessWillNeed expects data to be accessed in the near future.
	AccessWillNeed
	// AccessDontNeed expects data to not be accessed in the near future.
	AccessDontNeed
)

var (
	// ErrClosed is returned when attempting to access a closed mapping.
	ErrClosed = errors.New("mmap: mapping is closed")
	// ErrInvalidSize is returned when the file size is invalid (e.g. negative or too large).
	ErrInvalidSize = errors.New("mmap: invalid file size")
	// ErrOutOfBounds is returned when attempting to access a region outside the mapping.
	ErrOutOfBounds = errors.New("mmap: out of bounds")
	// ErrInvalidOffset is returned when the offset is invalid (e.g. negative).
	ErrInvalidOffset = errors.New("mmap: invalid offset")
)
