package resource

import (
	"context"
	"io"
)

// RateLimitedWriter wraps an io.Writer with rate limiting.
type RateLimitedWriter struct {
	w   io.Writer
	rc  *Controller
	ctx context.Context
}

// NewRateLimitedWriter creates a new RateLimitedWriter.
func NewRateLimitedWriter(w io.Writer, rc *Controller, ctx context.Context) *RateLimitedWriter {
	return &RateLimitedWriter{
		w:   w,
		rc:  rc,
		ctx: ctx,
	}
}

func (w *RateLimitedWriter) Write(p []byte) (n int, err error) {
	if err := w.rc.AcquireIO(w.ctx, len(p)); err != nil {
		return 0, err
	}
	return w.w.Write(p)
}

// RateLimitedReader wraps an io.Reader with rate limiting.
type RateLimitedReader struct {
	r   io.Reader
	rc  *Controller
	ctx context.Context
}

// NewRateLimitedReader creates a new RateLimitedReader.
func NewRateLimitedReader(r io.Reader, rc *Controller, ctx context.Context) *RateLimitedReader {
	return &RateLimitedReader{
		r:   r,
		rc:  rc,
		ctx: ctx,
	}
}

func (r *RateLimitedReader) Read(p []byte) (n int, err error) {
	// We don't know how many bytes we will read, but we can limit based on buffer size.
	// Or we can read first, then wait?
	// Better to wait for len(p) (max potential read) or a chunk.
	// If len(p) is huge, we might block too long.
	// Let's just wait for len(p).
	if err := r.rc.AcquireIO(r.ctx, len(p)); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}
