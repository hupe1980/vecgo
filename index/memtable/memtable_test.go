package memtable

import (
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemTable_InsertAndGet(t *testing.T) {
	m := New(3, distance.SquaredL2)

	vec1 := []float32{1.0, 2.0, 3.0}
	m.Insert(1, vec1)

	// Get existing
	vec, found, isDeleted := m.Get(1)
	assert.True(t, found)
	assert.False(t, isDeleted)
	assert.Equal(t, vec1, vec)

	// Get non-existing
	vec, found, isDeleted = m.Get(999)
	assert.False(t, found)
	assert.False(t, isDeleted)
	assert.Nil(t, vec)
}

func TestMemTable_Update(t *testing.T) {
	m := New(3, distance.SquaredL2)

	vec1 := []float32{1.0, 2.0, 3.0}
	m.Insert(1, vec1)

	vec2 := []float32{4.0, 5.0, 6.0}
	m.Insert(1, vec2)

	vec, found, isDeleted := m.Get(1)
	assert.True(t, found)
	assert.False(t, isDeleted)
	assert.Equal(t, vec2, vec)
}

func TestMemTable_Delete(t *testing.T) {
	m := New(3, distance.SquaredL2)

	// Delete existing
	vec1 := []float32{1.0, 2.0, 3.0}
	m.Insert(1, vec1)
	m.Delete(1)

	vec, found, isDeleted := m.Get(1)
	assert.True(t, found)
	assert.True(t, isDeleted)
	assert.Nil(t, vec)

	// Delete non-existing (Tombstone creation)
	m.Delete(2)
	vec, found, isDeleted = m.Get(2)
	assert.True(t, found)
	assert.True(t, isDeleted)
	assert.Nil(t, vec)
}

func TestMemTable_Search(t *testing.T) {
	m := New(2, distance.SquaredL2)

	// Insert vectors
	// Query: {0, 0}
	// ID 1: {1, 0} -> dist 1
	// ID 2: {2, 0} -> dist 4
	// ID 3: {0, 1} -> dist 1
	// ID 4: {10, 10} -> dist 200

	m.Insert(1, []float32{1.0, 0.0})
	m.Insert(2, []float32{2.0, 0.0})
	m.Insert(3, []float32{0.0, 1.0})
	m.Insert(4, []float32{10.0, 10.0})

	// Delete ID 3
	m.Delete(3)

	query := []float32{0.0, 0.0}
	results := m.Search(query, 2, nil)

	require.Len(t, results, 2)
	assert.Equal(t, uint64(1), results[0].ID)
	assert.InDelta(t, 1.0, results[0].Distance, 0.001)
	assert.Equal(t, uint64(2), results[1].ID)
	assert.InDelta(t, 4.0, results[1].Distance, 0.001)

	// Verify ID 3 is not returned (deleted)
	for _, res := range results {
		assert.NotEqual(t, uint64(3), res.ID)
	}
}

func TestMemTable_Search_Filter(t *testing.T) {
	m := New(2, distance.SquaredL2)

	m.Insert(1, []float32{1.0, 0.0})
	m.Insert(2, []float32{2.0, 0.0})

	query := []float32{0.0, 0.0}

	// Filter out ID 1
	filter := func(id uint64) bool {
		return id != 1
	}

	results := m.Search(query, 10, filter)
	require.Len(t, results, 1)
	assert.Equal(t, uint64(2), results[0].ID)
}

func TestMemTable_Flush(t *testing.T) {
	m := New(2, distance.SquaredL2)

	m.Insert(1, []float32{1.0, 0.0})
	m.Delete(2)

	assert.Equal(t, 2, m.Size())

	items := m.Flush()
	assert.Len(t, items, 2)
	assert.Equal(t, 0, m.Size())

	// Verify items
	idMap := make(map[uint64]Item)
	for _, item := range items {
		idMap[item.ID] = item
	}

	assert.False(t, idMap[1].IsDeleted)
	assert.Equal(t, []float32{1.0, 0.0}, idMap[1].Vector)

	assert.True(t, idMap[2].IsDeleted)
	assert.Nil(t, idMap[2].Vector)
}

func TestMemTable_Items(t *testing.T) {
	m := New(1, distance.SquaredL2)
	m.Insert(1, []float32{1.0})

	items := m.Items()
	assert.Len(t, items, 1)
	assert.Equal(t, 1, m.Size()) // Should not clear
}
