package manifest

import "errors"

var (
	// ErrIncompatibleVersion is returned when the manifest version is not supported.
	ErrIncompatibleVersion = errors.New("incompatible manifest version")

	// ErrNotFound is returned when the manifest file does not exist.
	ErrNotFound = errors.New("manifest not found")
)
