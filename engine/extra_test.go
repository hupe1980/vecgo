package engine

import (
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNoopMetricsObserver(t *testing.T) {
	o := &NoopMetricsObserver{}
	o.OnInsert(time.Second, nil)
	o.OnDelete(time.Second, nil)
	o.OnWALWrite(time.Second, 100)
	o.OnMemTableStatus(100, 0.5)
	o.OnBackpressure("memory")
	o.OnSearch(time.Second, "hnsw", 10, 10, nil)
	o.OnGet(time.Second, nil)
	o.OnFlush(time.Second, 100, 1000, nil)
	o.OnCompaction(time.Second, 10, 1, nil)
	o.OnBuild(time.Second, "hnsw", nil)
	o.OnStall(time.Second, "compaction")
	o.OnQueueDepth("queue", 5)
	o.OnThroughput("io", 100)
}

func TestTieredCompactionPolicy(t *testing.T) {
	p := &TieredCompactionPolicy{Threshold: 2}

	// Not enough segments
	segments := []SegmentStats{{ID: 1, Size: 100}}
	assert.Empty(t, p.Pick(segments))

	// Enough segments
	segments = []SegmentStats{{ID: 1, Size: 100}, {ID: 2, Size: 100}}
	picked := p.Pick(segments)
	assert.Len(t, picked, 2)
}

func TestSearchOptions(t *testing.T) {
	opts := &model.SearchOptions{}

	WithPreFilter(true)(opts)
	assert.NotNil(t, opts.PreFilter)
	assert.True(t, *opts.PreFilter)

	WithRefineFactor(2.0)(opts)
	assert.Equal(t, float32(2.0), opts.RefineFactor)

	WithNProbes(10)(opts)
	assert.Equal(t, 10, opts.NProbes)

	WithVector()(opts)
	assert.True(t, opts.IncludeVector)

	WithMetadata()(opts)
	assert.True(t, opts.IncludeMetadata)
}

func TestEngineOptionsCalls(t *testing.T) {
	e := &Engine{}

	WithResourceController(nil)(e)

	// DefaultWALOptions() returns WALOptions (struct)
	WithWALOptions(DefaultWALOptions())(e)

	WithCompactionPolicy(&TieredCompactionPolicy{})(e)
	WithBlobStore(nil)(e)
	WithBlockCacheSize(1024)(e)
}

func TestScanConfig(t *testing.T) {
	cfg := &ScanConfig{}
	WithScanBatchSize(100)(cfg)
	assert.Equal(t, 100, cfg.BatchSize)
}

func TestEngine_DebugInfo(t *testing.T) {
	dir := t.TempDir()
	e, err := Open(dir, 2, distance.MetricL2)
	require.NoError(t, err)
	defer e.Close()

	info := e.DebugInfo()
	assert.NotEmpty(t, info)
	assert.Contains(t, info, "Engine State:")
}
