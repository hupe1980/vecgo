package pk

import (
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
)

func TestPersistentIndex(t *testing.T) {
	idx := NewPersistentIndex()

	// 1. Insert
	idx1 := idx.Insert(model.PKUint64(1), model.Location{SegmentID: 1, RowID: 1})
	assert.Equal(t, 1, idx1.Len())
	loc, ok := idx1.Lookup(model.PKUint64(1))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 1, RowID: 1}, loc)

	// Original should be empty
	assert.Equal(t, 0, idx.Len())
	_, ok = idx.Lookup(model.PKUint64(1))
	assert.False(t, ok)

	// 2. Update
	idx2 := idx1.Insert(model.PKUint64(1), model.Location{SegmentID: 2, RowID: 2})
	assert.Equal(t, 1, idx2.Len())
	loc, ok = idx2.Lookup(model.PKUint64(1))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 2, RowID: 2}, loc)

	// idx1 should be unchanged
	loc, ok = idx1.Lookup(model.PKUint64(1))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 1, RowID: 1}, loc)

	// 3. Insert multiple
	idx3 := idx2.Insert(model.PKUint64(2), model.Location{SegmentID: 3, RowID: 3})
	assert.Equal(t, 2, idx3.Len())
	loc, ok = idx3.Lookup(model.PKUint64(1))
	assert.True(t, ok)
	loc, ok = idx3.Lookup(model.PKUint64(2))
	assert.True(t, ok)
	assert.Equal(t, model.Location{SegmentID: 3, RowID: 3}, loc)

	// 4. Delete
	idx4 := idx3.Delete(model.PKUint64(1))
	assert.Equal(t, 1, idx4.Len())
	_, ok = idx4.Lookup(model.PKUint64(1))
	assert.False(t, ok)
	loc, ok = idx4.Lookup(model.PKUint64(2))
	assert.True(t, ok)

	// idx3 unchanged
	assert.Equal(t, 2, idx3.Len())
	_, ok = idx3.Lookup(model.PKUint64(1))
	assert.True(t, ok)

	// 5. Collision / Deep structure
	// Insert many keys to force depth
	curr := idx
	n := 1000
	for i := 0; i < n; i++ {
		curr = curr.Insert(model.PKUint64(uint64(i)), model.Location{SegmentID: 1, RowID: model.RowID(i)})
	}
	assert.Equal(t, n, curr.Len())

	for i := 0; i < n; i++ {
		loc, ok := curr.Lookup(model.PKUint64(uint64(i)))
		assert.True(t, ok)
		assert.Equal(t, model.RowID(i), loc.RowID)
	}

	// Delete all
	for i := 0; i < n; i++ {
		curr = curr.Delete(model.PKUint64(uint64(i)))
	}
	assert.Equal(t, 0, curr.Len())
}
