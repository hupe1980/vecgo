package engine

import (
	"io"
)

// SeekableBuffer is an in-memory buffer that implements io.Writer, io.Seeker, and io.Reader.
// It's used for building segments in memory before uploading to cloud storage.
type SeekableBuffer struct {
	buf []byte
	pos int64
}

// NewSeekableBuffer creates a new SeekableBuffer.
func NewSeekableBuffer() *SeekableBuffer {
	return &SeekableBuffer{
		buf: make([]byte, 0, 1024*1024), // 1MB initial capacity
	}
}

// Write implements io.Writer.
func (b *SeekableBuffer) Write(p []byte) (n int, err error) {
	minCap := int(b.pos) + len(p)
	if minCap > cap(b.buf) {
		// Grow buffer
		newCap := cap(b.buf) * 2
		if newCap < minCap {
			newCap = minCap
		}
		newBuf := make([]byte, len(b.buf), newCap)
		copy(newBuf, b.buf)
		b.buf = newBuf
	}
	if minCap > len(b.buf) {
		b.buf = b.buf[:minCap]
	}
	n = copy(b.buf[b.pos:], p)
	b.pos += int64(n)
	return n, nil
}

// Seek implements io.Seeker.
func (b *SeekableBuffer) Seek(offset int64, whence int) (int64, error) {
	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = b.pos + offset
	case io.SeekEnd:
		newPos = int64(len(b.buf)) + offset
	default:
		return 0, io.ErrUnexpectedEOF
	}
	if newPos < 0 {
		return 0, io.ErrUnexpectedEOF
	}
	b.pos = newPos
	return newPos, nil
}

// Read implements io.Reader.
func (b *SeekableBuffer) Read(p []byte) (n int, err error) {
	if b.pos >= int64(len(b.buf)) {
		return 0, io.EOF
	}
	n = copy(p, b.buf[b.pos:])
	b.pos += int64(n)
	return n, nil
}

// Bytes returns the underlying byte slice.
func (b *SeekableBuffer) Bytes() []byte {
	return b.buf
}

// Len returns the length of the buffer.
func (b *SeekableBuffer) Len() int {
	return len(b.buf)
}

// Reset clears the buffer.
func (b *SeekableBuffer) Reset() {
	b.buf = b.buf[:0]
	b.pos = 0
}

// Sync is a no-op for in-memory buffers.
func (b *SeekableBuffer) Sync() error {
	return nil
}
