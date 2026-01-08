package memtable

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemTable_InsertWithPayload_Coverage(t *testing.T) {
	m, err := New(1, 2, distance.MetricL2, nil)
	require.NoError(t, err)
	defer m.DecRef()

	// 1. Normal Append
	pk1 := model.PKUint64(1)
	vec1 := []float32{1.0, 1.0}
	payload1 := []byte("payload1")
	id1, err := m.InsertWithPayload(pk1, vec1, nil, payload1)
	require.NoError(t, err)
	assert.Equal(t, model.RowID(0), id1)

	// Verify payload
	batch, err := m.Fetch(context.Background(), []uint32{uint32(id1)}, []string{"payload"})
	require.NoError(t, err)
	require.Equal(t, 1, batch.RowCount())
	assert.Equal(t, payload1, batch.Payload(0))

	assert.Greater(t, m.Size(), int64(0))

	// Test Advise
	assert.NoError(t, m.Advise(segment.AccessRandom))
}

func TestMemTable_EdgeCases(t *testing.T) {
	// Closed memtable
	m, err := New(1, 2, distance.MetricL2, nil)
	require.NoError(t, err)
	m.DecRef() // Close it

	_, err = m.Insert(model.PKUint64(1), []float32{1, 1})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "memtable is closed")

	// InsertWithPayload on closed
	_, err = m.InsertWithPayload(model.PKUint64(1), []float32{1, 1}, nil, nil)
	assert.Error(t, err)
}

func TestMemTable_Growth(t *testing.T) {
	// If we want to test growth, we can make a small initial capacity if New allowed it.
	// New hardcodes capacity to 1024.
	// We can try to insert 1025 items.
	// That might be slow but let's try 1050 items.

	m, err := New(1, 2, distance.MetricL2, nil)
	require.NoError(t, err)
	defer m.DecRef()

	count := 2000 // > 1024
	for i := 0; i < count; i++ {
		_, err := m.Insert(model.PKUint64(uint64(i)), []float32{0.1, 0.1})
		require.NoError(t, err)
	}
	assert.Equal(t, uint32(count), m.RowCount())
}
