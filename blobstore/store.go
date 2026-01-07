package blobstore

import (
	"context"
	"io"
	"os"
)

// ErrNotFound is returned when a blob does not exist.
//
// Implementations should return an error that satisfies `errors.Is(err, ErrNotFound)`.
// The default maps to `os.ErrNotExist`.
var ErrNotFound = os.ErrNotExist

// WritableBlob is a write-only handle to a data blob.
type WritableBlob interface {
	io.Writer
	io.Closer
	// Sync ensures the data is persisted to stable storage.
	Sync() error
}

// BlobStore is an abstraction for accessing immutable data blobs (segments).
// It supports both local file systems and remote object stores (S3, GCS).
type BlobStore interface {
	// Open opens a blob for reading.
	Open(ctx context.Context, name string) (Blob, error)
	// Create creates a new blob for writing.
	Create(ctx context.Context, name string) (WritableBlob, error)
	// Delete deletes a blob.
	Delete(ctx context.Context, name string) error
	// List returns all blobs matching the prefix.
	List(ctx context.Context, prefix string) ([]string, error)
}

// Blob is a read-only handle to a data blob.
type Blob interface {
	io.ReaderAt
	io.Closer
	// Size returns the size of the blob in bytes.
	Size() int64
	// ReadRange reads a range of bytes from the blob.
	// This makes it easier to optimize for range requests (e.g. S3 Range header).
	ReadRange(off, len int64) (io.ReadCloser, error)
}

// Mappable is an optional interface for Blobs that support memory mapping.
type Mappable interface {
	// Bytes returns the underlying byte slice.
	// The slice is valid until the Blob is closed.
	// This is a zero-copy operation if supported.
	Bytes() ([]byte, error)
}
