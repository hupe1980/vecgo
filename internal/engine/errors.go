package engine

import "errors"

var (
	// ErrClosed is returned when an operation is attempted on a closed engine or segment.
	ErrClosed = errors.New("engine closed")

	// ErrInvalidArgument is returned when an argument is invalid (e.g. wrong dimension, k <= 0).
	ErrInvalidArgument = errors.New("invalid argument")

	// ErrCorrupt is returned when data corruption is detected (checksum mismatch, etc.).
	ErrCorrupt = errors.New("data corruption detected")

	// ErrIncompatibleFormat is returned when the on-disk format is not supported.
	ErrIncompatibleFormat = errors.New("incompatible format")

	// ErrBackpressure is returned when the system is under heavy load and rejects the operation.
	ErrBackpressure = errors.New("backpressure: resource limit exceeded")

	// ErrNotFound is returned when a requested ID is not found.
	ErrNotFound = errors.New("not found")

	// ErrReadOnly is returned when a write operation is attempted on a read-only engine.
	// Use ReadOnly() option for stateless serverless deployments.
	ErrReadOnly = errors.New("engine is read-only")
)
