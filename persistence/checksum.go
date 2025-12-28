package persistence

import (
	"fmt"
	"hash"
	"hash/crc32"
	"io"
)

// Checksum utilities for snapshot integrity verification.
//
// Uses CRC32 (IEEE polynomial) for:
// - Fast computation (hardware-accelerated on modern CPUs)
// - Good error detection for storage corruption
// - Standard algorithm (well-tested, widely used)
//
// Note: CRC32 is NOT cryptographically secure. Do not use for
// tamper detection - only for detecting accidental corruption.

// CRC32Table is the IEEE polynomial table for checksum computation.
var CRC32Table = crc32.MakeTable(crc32.IEEE)

// CalculateChecksum calculates CRC32 checksum of data.
func CalculateChecksum(data []byte) uint32 {
	return crc32.ChecksumIEEE(data)
}

// ComputeChecksum computes CRC32 checksum of data (alias for CalculateChecksum).
func ComputeChecksum(data []byte) uint32 {
	return CalculateChecksum(data)
}

// ChecksumWriter wraps an io.Writer and computes a running CRC32 checksum.
type ChecksumWriter struct {
	w    io.Writer
	hash hash.Hash32
}

// NewChecksumWriter creates a new checksumming writer.
func NewChecksumWriter(w io.Writer) *ChecksumWriter {
	return &ChecksumWriter{
		w:    w,
		hash: crc32.New(CRC32Table),
	}
}

// Write implements io.Writer.
func (cw *ChecksumWriter) Write(p []byte) (int, error) {
	// Update checksum
	if _, err := cw.hash.Write(p); err != nil {
		return 0, err
	}
	// Write to underlying writer
	return cw.w.Write(p)
}

// Sum returns the current checksum value.
func (cw *ChecksumWriter) Sum() uint32 {
	return cw.hash.Sum32()
}

// Reset resets the checksum to initial state.
func (cw *ChecksumWriter) Reset() {
	cw.hash.Reset()
}

// ChecksumReader wraps an io.Reader and computes a running CRC32 checksum.
type ChecksumReader struct {
	r    io.Reader
	hash hash.Hash32
}

// NewChecksumReader creates a new checksumming reader.
func NewChecksumReader(r io.Reader) *ChecksumReader {
	return &ChecksumReader{
		r:    r,
		hash: crc32.New(CRC32Table),
	}
}

// Read implements io.Reader.
func (cr *ChecksumReader) Read(p []byte) (int, error) {
	n, err := cr.r.Read(p)
	if n > 0 {
		// Update checksum with bytes actually read
		if _, hashErr := cr.hash.Write(p[:n]); hashErr != nil {
			return n, hashErr
		}
	}
	return n, err
}

// Sum returns the current checksum value.
func (cr *ChecksumReader) Sum() uint32 {
	return cr.hash.Sum32()
}

// Reset resets the checksum to initial state.
func (cr *ChecksumReader) Reset() {
	cr.hash.Reset()
}

// Verify checks if the computed checksum matches the expected value.
func (cr *ChecksumReader) Verify(expected uint32) error {
	actual := cr.Sum()
	if actual != expected {
		return &ChecksumMismatchError{
			Expected: expected,
			Actual:   actual,
		}
	}
	return nil
}

// ChecksumMismatchError is returned when checksum verification fails.
type ChecksumMismatchError struct {
	Expected uint32
	Actual   uint32
}

func (e *ChecksumMismatchError) Error() string {
	return fmt.Sprintf("checksum mismatch: expected 0x%08x, got 0x%08x", e.Expected, e.Actual)
}

// IsChecksumMismatch returns true if err is a checksum mismatch error.
func IsChecksumMismatch(err error) bool {
	_, ok := err.(*ChecksumMismatchError)
	return ok
}
