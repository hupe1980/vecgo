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

func TestPersistentIndex_Scan(t *testing.T) {
	idx := NewPersistentIndex()

	data := []struct {
		k model.PK
		l model.Location
	}{
		{model.PKUint64(1), model.Location{SegmentID: 1, RowID: 1}},
		{model.PKString("two"), model.Location{SegmentID: 2, RowID: 2}},
		{model.PKUint64(3), model.Location{SegmentID: 3, RowID: 3}},
	}

	for _, d := range data {
		idx = idx.Insert(d.k, d.l)
	}

	count := 0
	found := make(map[model.PK]bool)
	for k, v := range idx.Scan() {
		count++
		found[k] = true
		// Verify value matches what we expect for this key
		switch {
		case k == data[0].k:
			assert.Equal(t, data[0].l, v)
		case k == data[1].k:
			assert.Equal(t, data[1].l, v)
		case k == data[2].k:
			assert.Equal(t, data[2].l, v)
		}
	}
	assert.Equal(t, 3, count)
	assert.Len(t, found, 3)
}

func TestPersistentIndex_Scan_EarlyExit(t *testing.T) {
	idx := NewPersistentIndex()
	for i := 0; i < 10; i++ {
		idx = idx.Insert(model.PKUint64(uint64(i)), model.Location{SegmentID: 1, RowID: 1})
	}

	count := 0
	for range idx.Scan() {
		count++
		if count == 5 {
			break
		}
	}
	assert.Equal(t, 5, count)
}
