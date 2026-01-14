// Package mmap provides memory-mapped file access for zero-copy I/O.
//
// # Overview
//
// Memory mapping allows direct access to file contents without copying data
// through kernel buffers. This is essential for high-performance vector search
// where segment files can be gigabytes in size.
//
// # Usage
//
//	m, err := mmap.Open("segment.bin")
//	if err != nil { ... }
//	defer m.Close()
//
//	// Zero-copy access to file contents
//	data := m.Bytes()
//
//	// Create a view into a specific region
//	region, _ := m.Region(offset, size)
//
//	// Provide kernel hints for access patterns
//	m.Advise(mmap.AccessSequential)
//
// # Platform Support
//
// The package provides a unified API across platforms:
//
//   - Unix (Linux, macOS, BSD): Uses mmap(2) with madvise(2) for access hints
//   - Windows: Uses CreateFileMapping/MapViewOfFile (madvise is a no-op)
//
// # Thread Safety
//
// Mapping and Region are safe for concurrent read access. The Close() method
// is idempotent and protected by atomic operations. However, callers must
// ensure no goroutines access Bytes() after Close() returns.
//
// # Anonymous Mappings
//
// MapAnon() creates read-write anonymous mappings for off-heap memory allocation.
// This is used by the Arena allocator to obtain large memory chunks outside
// the Go garbage collector's control.
package mmap
