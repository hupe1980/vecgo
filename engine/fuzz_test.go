package engine

import (
	"bytes"
	"io"
	"testing"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/internal/segment/flat"
	"github.com/hupe1980/vecgo/internal/wal"
)

var _ blobstore.Blob = (*bytesBlob)(nil)

type bytesBlob struct {
	data []byte
}

func (b *bytesBlob) ReadAt(p []byte, off int64) (n int, err error) {
	if off >= int64(len(b.data)) {
		return 0, io.EOF
	}
	n = copy(p, b.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (b *bytesBlob) Close() error           { return nil }
func (b *bytesBlob) Size() int64            { return int64(len(b.data)) }
func (b *bytesBlob) Bytes() ([]byte, error) { return b.data, nil }

func FuzzFlatSegmentOpen(f *testing.F) {
	f.Fuzz(func(t *testing.T, data []byte) {
		blob := &bytesBlob{data: data}
		// We expect Open to either succeed or return an error, but NOT panic.
		// We use WithVerifyChecksum(true) to exercise the checksum logic.
		s, err := flat.Open(blob, flat.WithVerifyChecksum(true))
		if err == nil {
			// If it opened successfully, we should be able to close it.
			// We can also try to read from it, but that might be too much for a simple fuzz test.
			_ = s.Close()
		}
	})
}

func FuzzWALDecode(f *testing.F) {
	f.Fuzz(func(t *testing.T, data []byte) {
		r := bytes.NewReader(data)
		// We expect Decode to either succeed or return an error, but NOT panic.
		_, _ = wal.Decode(r)
	})
}
