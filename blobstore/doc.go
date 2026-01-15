// Package blobstore provides storage abstraction for Vecgo's immutable segments.
//
// BlobStore is the interface for reading and writing data blobs (segments, manifests).
// Implementations must be safe for concurrent use.
//
// # Built-in Implementations
//
//   - LocalStore: Local filesystem with mmap support
//   - MemoryStore: In-memory store for testing
//   - s3.Store: Amazon S3 with range reads and parallel uploads
//   - s3.ExpressStore: S3 Express One Zone for low-latency access
//   - s3.DDBCommitStore: S3 + DynamoDB for concurrent writers
//   - minio.Store: MinIO and S3-compatible storage (native client)
//
// # Storage Selection Guide
//
//	Local Dev/Test:    blobstore.NewLocalStore(dir)
//	Unit Tests:        blobstore.NewMemoryStore()
//	AWS S3:            s3.NewStore(client, bucket, prefix)
//	S3 Express:        s3.NewExpressStore(client, bucket, prefix)
//	Multi-Writer:      s3.NewDDBCommitStore(s3Store, ddbClient, table, baseURI)
//	MinIO/Self-Hosted: minio.NewStore(client, bucket, prefix)
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
//	    ReadAt(ctx, p, off) (int, error)
//	    io.Closer
//	    Size() int64
//	    ReadRange(ctx, off, len) (io.ReadCloser, error)
//	}
package blobstore
