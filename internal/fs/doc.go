// Package fs provides filesystem abstractions for testability and fault injection.
//
// The package defines two key interfaces:
//
//   - [File]: Represents an open file with read/write/sync capabilities
//   - [FileSystem]: Abstracts filesystem operations (open, remove, rename, etc.)
//
// # Implementations
//
//   - [LocalFS]: Production implementation using standard os package
//   - [FaultyFS]: Test utility for fault injection (simulate I/O errors)
//
// # Usage
//
// Production code should use fs.Default (which is [LocalFS]):
//
//	file, err := fs.Default.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
//
// Tests can inject [FaultyFS] to simulate failures:
//
//	ffs := fs.NewFaultyFS(nil)
//	ffs.SetLimit(1024) // Fail after 1KB written
//	// inject ffs into component under test
//
// # Design Notes
//
// This package intentionally does NOT include context.Context parameters.
// Filesystem operations are typically fast (microseconds for local NVMe) and
// non-interruptible at the syscall level. Adding context would add overhead
// without meaningful cancellation capability.
//
// For slow operations (e.g., S3), use [blobstore.Blob] which has context support.
package fs
