package manifest

import (
	"bytes"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinaryRoundTrip(t *testing.T) {
	m := &Manifest{
		Version:       1,
		ID:            1,
		CreatedAt:     time.Now(),
		Dim:           2,
		Metric:        "L2",
		NextSegmentID: 2,
		MaxLSN:        123,
		Segments: []SegmentInfo{
			{
				ID:       1,
				Level:    0,
				RowCount: 10,
				Size:     1024,
				MinID:    1,
				MaxID:    10,
				Path:     "segment_1.bin",
			},
		},
		PKIndex: PKIndexInfo{Path: "pk.idx"},
	}

	var buf bytes.Buffer
	err := m.WriteBinary(&buf)
	require.NoError(t, err)
	t.Logf("Written %d bytes", buf.Len())

	m2, err := ReadBinary(&buf)
	require.NoError(t, err)

	assert.Equal(t, m.ID, m2.ID)
	assert.Equal(t, m.Dim, m2.Dim)
	assert.Equal(t, m.Metric, m2.Metric)
	assert.Equal(t, m.NextSegmentID, m2.NextSegmentID)
	assert.Equal(t, m.MaxLSN, m2.MaxLSN)
	assert.Equal(t, len(m.Segments), len(m2.Segments))
	
	if len(m2.Segments) > 0 {
		s := m2.Segments[0]
		assert.Equal(t, model.SegmentID(1), s.ID)
		assert.Equal(t, 0, s.Level)
		assert.Equal(t, uint32(10), s.RowCount)
		assert.Equal(t, int64(1024), s.Size)
		assert.Equal(t, model.ID(1), s.MinID)
		assert.Equal(t, model.ID(10), s.MaxID)
		assert.Equal(t, "segment_1.bin", s.Path)
	}
	assert.Equal(t, "pk.idx", m2.PKIndex.Path)
}
