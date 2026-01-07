package manifest

import "errors"

var (
	// ErrIncompatibleVersion is returned when the manifest version is not supported.
	ErrIncompatibleVersion = errors.New("incompatible manifest version")
)
