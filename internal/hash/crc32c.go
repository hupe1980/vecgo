package hash

import (
	"hash"
	"hash/crc32"
)

// crc32cTable is pre-computed for CRC32-Castagnoli polynomial.
// Computing this once avoids repeated MakeTable calls.
var crc32cTable = crc32.MakeTable(crc32.Castagnoli)

// CRC32C computes the CRC32-Castagnoli checksum of data.
// Uses hardware acceleration when available (SSE4.2, ARM CRC).
// Throughput: ~20 GB/s on modern x86, ~10 GB/s on ARM.
func CRC32C(data []byte) uint32 {
	return crc32.Checksum(data, crc32cTable)
}

// NewCRC32C returns a new CRC32-Castagnoli hash.Hash32.
// Uses hardware acceleration when available.
func NewCRC32C() hash.Hash32 {
	return crc32.New(crc32cTable)
}
