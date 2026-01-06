package wal

import (
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWAL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.wal")

	// 1. Write records
	w, err := Open(nil, path, DefaultOptions())
	require.NoError(t, err)

	recs := []*Record{
		{
			Type:     RecordTypeUpsert,
			PK:       model.PKUint64(1),
			Vector:   []float32{1.0, 2.0, 3.0},
			Metadata: []byte("meta1"),
		},
		{
			Type: RecordTypeDelete,
			PK:   model.PKUint64(2),
		},
		{
			Type:     RecordTypeUpsert,
			PK:       model.PKUint64(3),
			Vector:   []float32{4.0, 5.0, 6.0},
			Metadata: []byte("meta3"),
		},
	}

	for _, r := range recs {
		err := w.Append(r)
		require.NoError(t, err)
	}
	require.NoError(t, w.Close())

	// 2. Read records
	w2, err := Open(nil, path, DefaultOptions())
	require.NoError(t, err)
	defer w2.Close()

	reader, err := w2.Reader()
	require.NoError(t, err)
	defer reader.Close()

	var readRecs []*Record
	for {
		r, err := reader.Next()
		if err != nil {
			break
		}
		readRecs = append(readRecs, r)
	}

	assert.Equal(t, len(recs), len(readRecs))
	for i, r := range recs {
		assert.Equal(t, r.Type, readRecs[i].Type)
		assert.Equal(t, r.PK, readRecs[i].PK)
		if r.Type == RecordTypeUpsert {
			assert.Equal(t, r.Vector, readRecs[i].Vector)
			assert.Equal(t, r.Metadata, readRecs[i].Metadata)
		}
	}
}
