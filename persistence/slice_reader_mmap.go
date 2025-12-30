package persistence

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"unsafe"
)

// SliceReader provides bounds-checked reads from a byte slice.
// It is used by mmap loaders to avoid intermediate allocations.
type SliceReader struct {
	b   []byte
	off int
}

func NewSliceReader(b []byte) *SliceReader {
	return &SliceReader{b: b, off: 0}
}

func (r *SliceReader) Offset() int {
	if r == nil {
		return 0
	}
	return r.off
}

func (r *SliceReader) ReadBytes(n int) ([]byte, error) {
	if n < 0 || r.off+n > len(r.b) {
		return nil, fmt.Errorf("sliceReader: out of bounds read (%d bytes at %d, len=%d)", n, r.off, len(r.b))
	}
	out := r.b[r.off : r.off+n]
	r.off += n
	return out, nil
}

func (r *SliceReader) ReadUint32() (uint32, error) {
	b, err := r.ReadBytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *SliceReader) ReadUint16() (uint16, error) {
	b, err := r.ReadBytes(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b), nil
}

func (r *SliceReader) Remaining() []byte {
	if r.off >= len(r.b) {
		return nil
	}
	return r.b[r.off:]
}

func (r *SliceReader) Advance(n int) {
	r.off += n
}

func (r *SliceReader) ReadFileHeader() (*FileHeader, error) {
	sz := binary.Size(FileHeader{})
	if sz <= 0 {
		return nil, fmt.Errorf("sliceReader: invalid FileHeader size: %d", sz)
	}
	b, err := r.ReadBytes(sz)
	if err != nil {
		return nil, err
	}
	var h FileHeader
	if err := binary.Read(bytes.NewReader(b), binary.LittleEndian, &h); err != nil {
		return nil, err
	}
	if h.Magic != MagicNumber {
		return nil, fmt.Errorf("%w: got 0x%08x", ErrInvalidMagic, h.Magic)
	}
	if h.Version != Version {
		return nil, fmt.Errorf("%w: got 0x%08x", ErrInvalidVersion, h.Version)
	}
	return &h, nil
}

func (r *SliceReader) ReadUint32SliceCopy(n int) ([]uint32, error) {
	if n == 0 {
		return nil, nil
	}
	bb, err := r.ReadBytes(n * 4)
	if err != nil {
		return nil, err
	}
	out := make([]uint32, n)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), n*4), bb)
	return out, nil
}

func (r *SliceReader) ReadFloat32SliceView(n int) ([]float32, error) {
	if n == 0 {
		return nil, nil
	}
	bb, err := r.ReadBytes(n * 4)
	if err != nil {
		return nil, err
	}
	// bb comes from a mmapped region; we return a view into it.
	return unsafe.Slice((*float32)(unsafe.Pointer(&bb[0])), n), nil
}

func (r *SliceReader) ReadUint64() (uint64, error) {
	b, err := r.ReadBytes(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (r *SliceReader) ReadUint64SliceCopy(n int) ([]uint64, error) {
	if n == 0 {
		return nil, nil
	}
	bb, err := r.ReadBytes(n * 8)
	if err != nil {
		return nil, err
	}
	out := make([]uint64, n)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), n*8), bb)
	return out, nil
}
