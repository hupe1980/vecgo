package manifest

import "errors"

var (
	// ErrNotFound is returned when the manifest file does not exist.
	ErrNotFound = errors.New("manifest not found")

	// ErrInvalidManifest is returned when the manifest content is invalid.
	ErrInvalidManifest = errors.New("invalid manifest")
)
