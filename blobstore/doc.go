// Package blobstore provides storage abstraction for Vecgo's immutable segments.
//
// BlobStore is the interface for reading and writing data blobs (segments, manifests).
// Implementations must be safe for concurrent use.
//
// # Built-in Implementations
//
//   - LocalStore: Local filesystem with mmap support
//   - s3.Store: Amazon S3 with range reads and parallel uploads
//
// # Custom Implementations
//
// Implement the BlobStore interface to support custom storage backends:
//
//	type BlobStore interface {
//	    Open(ctx, name) (Blob, error)      // Open for reading
//	    Create(ctx, name) (WritableBlob, error)  // Create for writing
//	    Put(ctx, name, data) error         // Atomic write
//	    Delete(ctx, name) error
//	    List(ctx, prefix) ([]string, error)
//	}
//
// For cloud backends, implement ReadRange for efficient partial reads:
//
//	type Blob interface {
//	    io.ReaderAt
//	    io.Closer
//	    Size() int64
//	    ReadRange(off, len int64) (io.ReadCloser, error)
//	}
package blobstore
