package memtable

import (
	"context"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemTable_BasicCRUD(t *testing.T) {
	mt, err := New(1, 4, distance.MetricL2, nil)
	require.NoError(t, err)

	// Check Initial State
	assert.Equal(t, model.SegmentID(1), mt.ID())
	assert.Equal(t, uint32(0), mt.RowCount())
	assert.Equal(t, distance.MetricL2, mt.Metric())

	// Insert
	id1User := model.ID(16) // Shard 0 (16%16=0)
	vec1 := []float32{1, 0, 0, 0}
	md1 := metadata.Document{"tag": metadata.String("a")}
	payload1 := []byte("payload1")

	id1, err := mt.InsertWithPayload(id1User, vec1, md1, payload1)
	require.NoError(t, err)
	// Shard 0, Row 0 -> Global RowID 0 (if bitmask is 0)
	// Just verify it works
	assert.Equal(t, uint32(1), mt.RowCount())

	id2User := model.ID(17) // Shard 1 (17%16=1)
	vec2 := []float32{0, 1, 0, 0}
	id2, err := mt.Insert(id2User, vec2) // Insert without payload/md
	require.NoError(t, err)
	// Different shards -> different RowIDs
	assert.NotEqual(t, id1, id2)
	assert.Equal(t, uint32(2), mt.RowCount())

	// Fetch Full using returned IDs
	batch, err := mt.Fetch(context.Background(), []uint32{uint32(id1), uint32(id2)}, nil)
	require.NoError(t, err)
	require.Equal(t, 2, batch.RowCount())
	assert.Equal(t, id1User, batch.ID(0))
	assert.Equal(t, id2User, batch.ID(1))
	assert.Equal(t, vec1, batch.Vector(0))
	assert.Equal(t, vec2, batch.Vector(1))
	assert.Equal(t, "payload1", string(batch.Payload(0)))
	assert.Nil(t, batch.Payload(1))
	assert.Equal(t, "a", batch.Metadata(0)["tag"].StringValue())
	assert.Nil(t, batch.Metadata(1))

	// Fetch Columns
	batchCols, err := mt.Fetch(context.Background(), []uint32{uint32(id1)}, []string{"vector", "payload"})
	require.NoError(t, err)
	assert.Equal(t, vec1, batchCols.Vector(0))
	assert.Equal(t, "payload1", string(batchCols.Payload(0)))
	assert.Nil(t, batchCols.Metadata(0)) // Should be nil as not requested

	// Fetch IDs Optimized
	ids := make([]model.ID, 2)
	err = mt.FetchIDs(context.Background(), []uint32{uint32(id1), uint32(id2)}, ids)
	require.NoError(t, err)
	assert.Equal(t, id1User, ids[0])
	assert.Equal(t, id2User, ids[1])

	// Iterate
	count := 0
	err = mt.Iterate(func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error {
		count++
		if id == id1User {
			assert.Equal(t, vec1, vec)
			assert.Equal(t, "payload1", string(payload))
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, count)

	// Delete
	mt.Delete(id1)

	// Check Ref Counting Close
	// Start with 1 ref.
	mt.IncRef() // 2
	mt.DecRef() // 1
	mt.DecRef() // 0 -> Close

	// Operations after close should fail
	_, err = mt.Insert(id1User, vec1)
	assert.Error(t, err)
	_, err = mt.Fetch(context.Background(), []uint32{0}, nil)
	assert.Error(t, err)
}

func TestMemTable_Search(t *testing.T) {
	mt, err := New(1, 2, distance.MetricL2, nil)
	require.NoError(t, err)
	defer mt.Close()

	// Points: (0,0), (1,1), (2,2), (3,3)
	// IDs: 0, 16, 32, 48 -> All Shard 0 (to simulate collision density)
	// OR IDs: 0, 1, 2, 3 -> Shards 0, 1, 2, 3
	// Let's use scattered to test sharding
	ids := []model.ID{0, 1, 2, 3}
	rowIDs := make([]model.RowID, 4)

	for i := 0; i < 4; i++ {
		v := float32(i)
		rid, err := mt.Insert(ids[i], []float32{v, v})
		require.NoError(t, err)
		rowIDs[i] = rid
	}

	ctx := context.Background()
	q := []float32{0.1, 0.1} // Closest to (0,0)

	s := searcher.Get()
	defer searcher.Put(s)
	s.Heap.Reset(false)

	// Search k=2
	err = mt.Search(ctx, q, 2, nil, model.SearchOptions{}, s)
	require.NoError(t, err)

	cands := s.Heap.GetCandidates()
	assert.Len(t, cands, 2)

	// Best is 0 (score ~0), Next is 1 (score ~2).
	// Heap is MaxHeap (Descending). So Worst is top.
	// cands[0] should be 1. cands[1] should be 0.

	assert.Equal(t, rowIDs[1], model.RowID(cands[0].RowID))
	assert.Equal(t, rowIDs[0], model.RowID(cands[1].RowID))

	// Delete 0 and check search
	mt.Delete(rowIDs[0])
	s.Heap.Reset(false)
	err = mt.Search(ctx, q, 2, nil, model.SearchOptions{}, s)
	require.NoError(t, err)
	cands = s.Heap.GetCandidates()
	// Should now be 1 and 2
	assert.Equal(t, 2, len(cands))
	// 2 is worse than 1.
	assert.Equal(t, rowIDs[2], model.RowID(cands[0].RowID))
	assert.Equal(t, rowIDs[1], model.RowID(cands[1].RowID))
}

func TestMemTable_Rerank(t *testing.T) {
	mt, err := New(1, 2, distance.MetricL2, nil)
	require.NoError(t, err)
	defer mt.Close()

	rid, _ := mt.Insert(model.ID(16), []float32{0, 0}) // Shard 0

	cands := []model.Candidate{
		{Loc: model.Location{SegmentID: 1, RowID: rid}, Approx: true},          // Good
		{Loc: model.Location{SegmentID: 1, RowID: rid + 999999}, Approx: true}, // Bad RowID (Overflow/invalid)
		{Loc: model.Location{SegmentID: 99, RowID: rid}, Approx: true},         // Bad SegmentID
	}

	q := []float32{1, 0} // Dist to (0,0) is 1.0 (sq is 1)

	res, err := mt.Rerank(context.Background(), q, cands, nil)
	require.NoError(t, err)

	assert.Len(t, res, 1)
	assert.Equal(t, rid, res[0].Loc.RowID)
	assert.False(t, res[0].Approx)
	assert.InDelta(t, 1.0, res[0].Score, 0.001)
}

func TestMemTable_Concurrency(t *testing.T) {
	mt, err := New(1, 128, distance.MetricL2, nil)
	require.NoError(t, err)
	defer mt.Close()

	var wg sync.WaitGroup
	workers := 10
	ops := 100

	// Concurrent Inserts
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id)))
			vec := make([]float32, 128)
			for i := 0; i < ops; i++ {
				for j := range vec {
					vec[j] = rng.Float32()
				}
				_, err := mt.Insert(model.ID(uint64(id*ops+i)), vec)
				assert.NoError(t, err)
			}
		}(w)
	}

	// Concurrent Reads
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(10 * time.Millisecond) // Wait for some data
			s := searcher.Get()
			defer searcher.Put(s)
			s.Heap.Reset(false)

			q := make([]float32, 128)
			_ = mt.Search(context.Background(), q, 5, nil, model.SearchOptions{}, s)

			// RWMutex should protect this
			_ = mt.RowCount()
		}()
	}

	wg.Wait()
	assert.Equal(t, uint32(workers*ops), mt.RowCount())
}

func TestMemTable_Errors(t *testing.T) {
	_, err := New(1, 0, distance.MetricL2, nil)
	assert.Error(t, err)

	mt, _ := New(1, 2, distance.MetricL2, nil)
	mt.Close() // Should close immediately

	_, err = mt.Insert(model.ID(1), []float32{1, 1})
	assert.Error(t, err)

	err = mt.Search(context.Background(), nil, 10, nil, model.SearchOptions{}, nil)
	assert.Error(t, err)

	_, err = mt.Rerank(context.Background(), nil, []model.Candidate{{}}, nil)
	assert.Error(t, err)

	_, err = mt.Fetch(context.Background(), []uint32{0}, nil)
	assert.Error(t, err)

	err = mt.FetchIDs(context.Background(), []uint32{0}, nil)
	assert.Error(t, err)

	err = mt.Iterate(nil)
	assert.Error(t, err)
}
