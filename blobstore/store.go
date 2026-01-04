package blobstore

import (
	"io"
	"os"
)

// ErrNotFound is returned when a blob does not exist.
//
// Implementations should return an error that satisfies `errors.Is(err, ErrNotFound)`.
// The default maps to `os.ErrNotExist`.
var ErrNotFound = os.ErrNotExist

// BlobStore is an abstraction for accessing immutable data blobs (segments).
type BlobStore interface {
	// Open opens a blob for reading.
	Open(name string) (Blob, error)
}

// Blob is a read-only handle to a data blob.
type Blob interface {
	io.ReaderAt
	io.Closer
	// Size returns the size of the blob in bytes.
	Size() int64
}

// Mappable is an optional interface for Blobs that support memory mapping.
type Mappable interface {
	// Bytes returns the underlying byte slice.
	// The slice is valid until the Blob is closed.
	// This is a zero-copy operation if supported.
	Bytes() ([]byte, error)
}
