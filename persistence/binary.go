// Package persistence provides high-performance binary serialization for vector indexes.
// This replaced a slower, reflection-heavy encoding used in earlier iterations.
package persistence

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unsafe"
)

// BinaryIndexWriter writes indexes in optimized binary format.
type BinaryIndexWriter struct {
	w         io.Writer
	byteOrder binary.ByteOrder
	checksum  uint32
}

// NewBinaryIndexWriter creates a new binary writer.
func NewBinaryIndexWriter(w io.Writer) *BinaryIndexWriter {
	return &BinaryIndexWriter{
		w:         w,
		byteOrder: binary.LittleEndian, // Native on x86/ARM
	}
}

// WriteHeader writes the file header.
func (bw *BinaryIndexWriter) WriteHeader(header *FileHeader) error {
	header.Magic = MagicNumber
	header.Version = Version
	return binary.Write(bw.w, bw.byteOrder, header)
}

// WriteFloat32Slice writes a float32 slice as raw bytes (zero-copy compatible).
// Safety: Validates alignment before unsafe conversion.
func (bw *BinaryIndexWriter) WriteFloat32Slice(vec []float32) error {
	if len(vec) == 0 {
		return nil
	}

	// Verify alignment before unsafe operation
	if err := validateFloat32SliceAlignment(vec); err != nil {
		return err
	}

	// Direct memory conversion (no allocation)
	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&vec[0])), len(vec)*4)
	_, err := bw.w.Write(byteSlice)
	return err
}

// WriteUint32Slice writes a uint32 slice as raw bytes.
// Safety: Validates alignment before unsafe conversion.
func (bw *BinaryIndexWriter) WriteUint32Slice(slice []uint32) error {
	if len(slice) == 0 {
		return nil
	}

	// Verify alignment before unsafe operation
	if err := validateUint32SliceAlignment(slice); err != nil {
		return err
	}

	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&slice[0])), len(slice)*4)
	_, err := bw.w.Write(byteSlice)
	return err
}

// WriteUint64Slice writes a uint64 slice as raw bytes.
// Safety: Validates alignment before unsafe conversion.
func (bw *BinaryIndexWriter) WriteUint64Slice(slice []uint64) error {
	if len(slice) == 0 {
		return nil
	}

	// Verify alignment before unsafe operation
	if err := validateUint64SliceAlignment(slice); err != nil {
		return err
	}

	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&slice[0])), len(slice)*8)
	_, err := bw.w.Write(byteSlice)
	return err
}

// BinaryIndexReader reads indexes from binary format.
type BinaryIndexReader struct {
	r         io.Reader
	byteOrder binary.ByteOrder
}

// NewBinaryIndexReader creates a new binary reader.
func NewBinaryIndexReader(r io.Reader) *BinaryIndexReader {
	return &BinaryIndexReader{
		r:         r,
		byteOrder: binary.LittleEndian,
	}
}

// ReadHeader reads and validates the file header.
func (br *BinaryIndexReader) ReadHeader() (*FileHeader, error) {
	var header FileHeader
	if err := binary.Read(br.r, br.byteOrder, &header); err != nil {
		return nil, err
	}
	if header.Magic != MagicNumber {
		return nil, fmt.Errorf("%w: got 0x%08x", ErrInvalidMagic, header.Magic)
	}
	if header.Version != Version {
		return nil, fmt.Errorf("%w: got 0x%08x", ErrInvalidVersion, header.Version)
	}
	return &header, nil
}

// ReadFloat32Slice reads a float32 slice.
func (br *BinaryIndexReader) ReadFloat32Slice(count int) ([]float32, error) {
	if count == 0 {
		return nil, nil
	}
	vec := make([]float32, count)
	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&vec[0])), count*4)
	if _, err := io.ReadFull(br.r, byteSlice); err != nil {
		return nil, err
	}
	return vec, nil
}

// ReadFloat32SliceInto reads a float32 slice into the provided buffer.
func (br *BinaryIndexReader) ReadFloat32SliceInto(vec []float32) error {
	if len(vec) == 0 {
		return nil
	}
	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&vec[0])), len(vec)*4)
	if _, err := io.ReadFull(br.r, byteSlice); err != nil {
		return err
	}
	return nil
}

// ReadUint32Slice reads a uint32 slice.
func (br *BinaryIndexReader) ReadUint32Slice(count int) ([]uint32, error) {
	if count == 0 {
		return nil, nil
	}
	slice := make([]uint32, count)
	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&slice[0])), count*4)
	if _, err := io.ReadFull(br.r, byteSlice); err != nil {
		return nil, err
	}
	return slice, nil
}

// ReadUint64Slice reads a uint64 slice.
func (br *BinaryIndexReader) ReadUint64Slice(count int) ([]uint64, error) {
	if count == 0 {
		return nil, nil
	}
	slice := make([]uint64, count)
	byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&slice[0])), count*8)
	if _, err := io.ReadFull(br.r, byteSlice); err != nil {
		return nil, err
	}
	return slice, nil
}

// SaveToFile is a helper to save data to a file.
func SaveToFile(filename string, writeFunc func(io.Writer) error) error {
	dir := filepath.Dir(filename)
	base := filepath.Base(filename)

	// Write to a temp file in the same directory to ensure rename is atomic.
	tmp, err := os.CreateTemp(dir, base+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}()

	// Match typical file permissions (best-effort).
	_ = tmp.Chmod(0644)

	// Use buffered writer to batch writes (critical for performance)
	buf := bufio.NewWriterSize(tmp, 256*1024) // 256KB buffer
	if err := writeFunc(buf); err != nil {
		return err
	}
	if err := buf.Flush(); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	// Atomically replace target.
	if err := os.Rename(tmpName, filename); err != nil {
		return err
	}

	// Best-effort: fsync the directory so the rename is durable on POSIX.
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}

	// Success: prevent deferred cleanup from removing the final file.
	tmpName = ""
	return nil
}

// LoadFromFile is a helper to load data from a file.
func LoadFromFile(filename string, readFunc func(io.Reader) error) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Use buffered reader to batch reads
	buf := bufio.NewReaderSize(f, 256*1024) // 256KB buffer
	return readFunc(buf)
}
