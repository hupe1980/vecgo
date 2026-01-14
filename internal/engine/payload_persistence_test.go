package engine

import (
	"context"
	"os"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/require"
)

func TestPayloadSurvivesFlushAndCompaction(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	e, err := OpenLocal(context.Background(), dir, WithDimension(2), WithMetric(distance.MetricL2))
	require.NoError(t, err)
	defer e.Close()

	id1, err := e.Insert(context.Background(), []float32{1, 0}, nil, []byte("p1"))
	require.NoError(t, err)
	require.NoError(t, e.Commit(context.Background()))

	res, err := e.Search(ctx, []float32{1, 0}, 1, WithPayload())
	require.NoError(t, err)
	require.Len(t, res, 1)
	require.Equal(t, id1, res[0].ID)
	require.Equal(t, []byte("p1"), res[0].Payload)

	require.NoError(t, e.Close())

	e2, err := OpenLocal(context.Background(), dir, WithDimension(2), WithMetric(distance.MetricL2))
	require.NoError(t, err)
	defer e2.Close()

	res, err = e2.Search(ctx, []float32{1, 0}, 1, WithPayload())
	require.NoError(t, err)
	require.Len(t, res, 1)
	require.Equal(t, id1, res[0].ID)
	require.Equal(t, []byte("p1"), res[0].Payload)

	id2, err := e2.Insert(ctx, []float32{0, 1}, nil, []byte("p2"))
	require.NoError(t, err)
	require.NoError(t, e2.Commit(ctx))

	snap := e2.current.Load()
	segmentsToCompact := make([]model.SegmentID, 0, len(snap.segments))
	for id := range snap.segments {
		segmentsToCompact = append(segmentsToCompact, id)
	}
	require.Len(t, segmentsToCompact, 2)

	require.NoError(t, e2.Compact(segmentsToCompact, 1))

	res, err = e2.Search(ctx, []float32{1, 0}, 1, WithPayload())
	require.NoError(t, err)
	require.Len(t, res, 1)
	require.Equal(t, id1, res[0].ID)
	require.Equal(t, []byte("p1"), res[0].Payload)

	res, err = e2.Search(ctx, []float32{0, 1}, 1, WithPayload())
	require.NoError(t, err)
	require.Len(t, res, 1)
	require.Equal(t, id2, res[0].ID)
	require.Equal(t, []byte("p2"), res[0].Payload)
}

func TestOpenIgnoresOrphanTmpFiles(t *testing.T) {
	dir := t.TempDir()

	e, err := OpenLocal(context.Background(), dir, WithDimension(2), WithMetric(distance.MetricL2))
	require.NoError(t, err)

	_, err = e.Insert(context.Background(), []float32{1, 0}, nil, []byte("p1"))
	require.NoError(t, err)
	require.NoError(t, e.Commit(context.Background()))
	require.NoError(t, e.Close())

	// Create orphan tmp artifacts (simulates crash mid-write).
	require.NoError(t, os.WriteFile(dir+"/segment_999.bin.tmp", []byte("junk"), 0o644))
	require.NoError(t, os.WriteFile(dir+"/segment_999.payload.tmp", []byte("junk"), 0o644))
	require.NoError(t, os.WriteFile(dir+"/segment_999.tmp", []byte("junk"), 0o644))

	e2, err := OpenLocal(context.Background(), dir, WithDimension(2), WithMetric(distance.MetricL2))
	require.NoError(t, err)
	require.NoError(t, e2.Close())
}
