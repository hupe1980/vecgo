// Package hash provides fast, hardware-accelerated hashing utilities for data integrity.
//
// # CRC32-Castagnoli (CRC32C)
//
// All checksums in Vecgo use CRC32-Castagnoli (CRC32C) which provides:
//
//   - Hardware acceleration on x86 (SSE4.2) and ARM (CRC extension)
//   - 10-20 GB/s throughput on modern CPUs
//   - Superior error detection compared to CRC32-IEEE
//   - Industry standard (iSCSI, Btrfs, RocksDB, LevelDB)
//
// The CRC32C polynomial (0x1EDC6F41) detects all single-bit, double-bit, and
// odd-bit errors, plus burst errors up to 32 bits with 100% reliability.
//
// # Usage
//
// For one-shot checksums:
//
//	checksum := hash.CRC32C(data)
//
// For streaming checksums:
//
//	h := hash.NewCRC32C()
//	h.Write(chunk1)
//	h.Write(chunk2)
//	checksum := h.Sum32()
//
// # Performance
//
// The crc32cTable is pre-computed at package init time, avoiding repeated
// table generation. Go's crc32 package automatically uses hardware
// instructions when available:
//
//	Platform          Throughput
//	x86-64 (SSE4.2)   ~20 GB/s
//	ARM64 (CRC)       ~10 GB/s
//	Software          ~2 GB/s
package hash
