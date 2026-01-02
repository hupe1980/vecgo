package hnsw

import (
	"bytes"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/persistence"
)

func init() {
	index.RegisterMmapBinaryLoader(persistence.IndexTypeHNSW, func(data []byte) (index.Index, int, error) {
		h := &HNSW{}
		// We can use ReadFromWithOptions with a bytes.Reader
		// This is not true mmap (zero-copy), but it satisfies the interface.
		// The consumed bytes will be calculated by the reader.

		r := bytes.NewReader(data)
		if err := h.ReadFromWithOptions(r, DefaultOptions); err != nil {
			return nil, 0, err
		}

		// Calculate consumed bytes
		consumed := len(data) - r.Len()
		return h, consumed, nil
	})
}
