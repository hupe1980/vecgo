//go:build amd64 || arm64

// Package persistence provides high-performance binary serialization for vector indexes.
//
// PLATFORM REQUIREMENTS:
// - Architecture: amd64 or arm64 only
// - Endianness: Little-endian (native on x86_64 and ARM64)
// - Alignment: 4-byte for float32/uint32, 8-byte for uint64
//
// The unsafe operations in this package are verified at runtime with alignment checks
// and platform validation. See safety.go for implementation details.
package persistence
