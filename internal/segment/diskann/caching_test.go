package diskann

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockCache struct {
	data map[cache.CacheKey][]byte
	gets int
	sets int
}

func (m *mockCache) Get(ctx context.Context, key cache.CacheKey) ([]byte, bool) {
	m.gets++
	b, ok := m.data[key]
	return b, ok
}

func (m *mockCache) Set(ctx context.Context, key cache.CacheKey, b []byte) {
	m.sets++
	m.data[key] = b
}

func (m *mockCache) Invalidate(predicate func(key cache.CacheKey) bool) {}

func (m *mockCache) Close() error { return nil }

func (m *mockCache) Stats() (hits, misses int64) { return 0, 0 }

func TestCachingIntegration(t *testing.T) {
	// Create a valid segment file
	h := FileHeader{
		Magic:        MagicNumber,
		Version:      Version,
		RowCount:     2,
		Dim:          4,
		MaxDegree:    2,
		PKOffset:     HeaderSize,
		VectorOffset: HeaderSize + 16,               // 2 rows * 8 bytes
		GraphOffset:  HeaderSize + 16 + (2 * 4 * 4), // Vectors end
	}

	// Create data
	buf := h.Encode()
	// Pad to PKOffset
	buf = append(buf, make([]byte, h.PKOffset-uint64(len(buf)))...)

	// PKs: 1, 2
	pkBuf := make([]byte, 16)
	binary.LittleEndian.PutUint64(pkBuf[0:], 1)
	binary.LittleEndian.PutUint64(pkBuf[8:], 2)
	buf = append(buf, pkBuf...)

	// Vectors: [1.0 ...], [2.0 ...]
	vecBuf := make([]byte, 2*4*4)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(vecBuf[i*4:], math.Float32bits(1.0))
	}
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(vecBuf[16+i*4:], math.Float32bits(2.0))
	}
	buf = append(buf, vecBuf...)

	// Graph: [0, 0], [0, 0]
	graphBuf := make([]byte, 2*2*4)
	buf = append(buf, graphBuf...)

	blob := &nonMappableBlob{data: buf}
	c := &mockCache{data: make(map[cache.CacheKey][]byte)}

	s, err := Open(blob, WithBlockCache(c))
	require.NoError(t, err)
	defer s.Close()

	// 1. Read Vector suitable for cache (Lazy load)
	// access row 0
	vec, err := s.Get(0)
	require.NoError(t, err)
	assert.Equal(t, float32(1.0), vec[0])

	// Should satisfy from blob, and SET to cache
	assert.Equal(t, 1, c.sets) // 1 page (4KB) containing header + data
	assert.Equal(t, 1, c.gets) // Get is called first (miss)

	// Technically Get is called, returns false (miss).
	// But in my mock, I increment gets on call.
	// Let's re-read Segment.readBlock logic:
	// if pageData, ok = s.cache.Get(...); !ok { ... s.cache.Set(...) }

	// Correction: gets should be 1 (miss), sets should be 1.
	// Wait, vector offset is small, so it falls in page 0.
	// BUT, Open might read header. Open reads header directly using `s.blob.ReadAt` in `load()`.
	// Only `s.Get()` calls `readVector` -> `readBlock`.

	// Let's verify counts.
	// s.Get(0) calls readVector.
	// readVector calls readBlock.
	// readBlock calculates page. Offset ~ 200 bytes. Page 0.
	// Cache.Get(Page 0) -> Miss.
	// Cache.Set(Page 0).

	// 2. Read Again
	c.sets = 0
	c.gets = 0
	vec, err = s.Get(0)
	require.NoError(t, err)
	assert.Equal(t, float32(1.0), vec[0])

	assert.Equal(t, 1, c.gets) // Hit
	assert.Equal(t, 0, c.sets)

	// 3. Read neighbor (Graph)
	// Graph offset is also small, likely page 0.
	// So it should hit cache.
	// readGraphNode(0) -> readBlock
	// Page 0 -> Hit.
	c.gets = 0
	_, err = s.readGraphNode(0)
	require.NoError(t, err)
	assert.Equal(t, 1, c.gets)
}
