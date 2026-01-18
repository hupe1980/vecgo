package benchmark_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

// ============================================================================
// Fixture Configuration
// ============================================================================

// FixtureConfig defines a benchmark fixture's parameters.
type FixtureConfig struct {
	Name         string
	Dim          int
	NumVecs      int
	BucketCount  int
	Distribution string
	ZipfSkew     float64
	VectorNoise  float32
	MissingRate  float64
	Seed         int64
}

// StandardFixtures defines the canonical set of benchmark fixtures.
var StandardFixtures = []FixtureConfig{
	// Small fixtures for CI
	{Name: "uniform_128d_10k", Dim: 128, NumVecs: 10_000, BucketCount: 100, Distribution: "uniform", Seed: 42},
	{Name: "zipfian_128d_10k", Dim: 128, NumVecs: 10_000, BucketCount: 100, Distribution: "zipfian", ZipfSkew: 1.5, VectorNoise: 0.15, MissingRate: 0.30, Seed: 42},

	// Medium fixtures
	{Name: "uniform_128d_50k", Dim: 128, NumVecs: 50_000, BucketCount: 100, Distribution: "uniform", Seed: 42},
	{Name: "zipfian_128d_50k", Dim: 128, NumVecs: 50_000, BucketCount: 100, Distribution: "zipfian", ZipfSkew: 1.5, VectorNoise: 0.15, MissingRate: 0.30, Seed: 42},
	{Name: "nofilter_128d_50k", Dim: 128, NumVecs: 50_000, BucketCount: 0, Distribution: "nofilter", Seed: 42},

	// Production-scale
	{Name: "uniform_768d_50k", Dim: 768, NumVecs: 50_000, BucketCount: 100, Distribution: "uniform", Seed: 42},
	{Name: "zipfian_768d_50k", Dim: 768, NumVecs: 50_000, BucketCount: 100, Distribution: "zipfian", ZipfSkew: 1.5, VectorNoise: 0.15, MissingRate: 0.30, Seed: 42},

	// Large fixtures
	{Name: "uniform_768d_100k", Dim: 768, NumVecs: 100_000, BucketCount: 100, Distribution: "uniform", Seed: 42},
	{Name: "zipfian_768d_100k", Dim: 768, NumVecs: 100_000, BucketCount: 100, Distribution: "zipfian", ZipfSkew: 1.5, VectorNoise: 0.15, MissingRate: 0.30, Seed: 42},
}

// QuickFixtures are fast fixtures for CI.
var QuickFixtures = []FixtureConfig{
	{Name: "uniform_128d_10k", Dim: 128, NumVecs: 10_000, BucketCount: 100, Distribution: "uniform", Seed: 42},
	{Name: "zipfian_128d_10k", Dim: 128, NumVecs: 10_000, BucketCount: 100, Distribution: "zipfian", ZipfSkew: 1.5, VectorNoise: 0.15, MissingRate: 0.30, Seed: 42},
}

// ============================================================================
// Fixture Storage
// ============================================================================

const (
	fixtureBaseDir  = "testdata/fixtures"
	fixtureMetaFile = "fixture.meta"
	queryFile       = "queries.bin"
	groundTruthFile = "ground_truth.bin"
)

// FixtureDir returns the directory path for a fixture.
func FixtureDir(name string) string {
	return filepath.Join(fixtureBaseDir, name)
}

// FixtureExists checks if a fixture has been generated.
func FixtureExists(name string) bool {
	metaPath := filepath.Join(FixtureDir(name), fixtureMetaFile)
	_, err := os.Stat(metaPath)
	return err == nil
}

// ============================================================================
// Fixture Data
// ============================================================================

// FixtureData holds pre-computed benchmark data.
type FixtureData struct {
	Config      FixtureConfig
	Queries     [][]float32
	GroundTruth map[string][][]model.ID
	Buckets     []int64
	Present     []bool
	PKs         []model.ID
}

var (
	fixtureCache   = make(map[string]*FixtureData)
	fixtureCacheMu sync.RWMutex
)

// ============================================================================
// Fixture Generation
// ============================================================================

// GenerateFixture creates a persistent benchmark fixture.
func GenerateFixture(cfg FixtureConfig) error {
	dir := FixtureDir(cfg.Name)

	if err := os.RemoveAll(dir); err != nil {
		return fmt.Errorf("remove old fixture: %w", err)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create fixture dir: %w", err)
	}

	ctx := context.Background()

	db, err := vecgo.Open(ctx, vecgo.Local(dir),
		vecgo.Create(cfg.Dim, vecgo.MetricL2),
		vecgo.WithCompactionThreshold(1<<40),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
		vecgo.WithDiskANNThreshold(cfg.NumVecs+1),
		vecgo.WithMemoryLimit(0),
	)
	if err != nil {
		return fmt.Errorf("open db: %w", err)
	}
	defer db.Close()

	rng := testutil.NewRNG(cfg.Seed)

	var data [][]float32
	var buckets []int64
	var present []bool

	switch cfg.Distribution {
	case "uniform":
		data, buckets = generateUniformData(rng, cfg)
		present = make([]bool, cfg.NumVecs)
		for i := range present {
			present[i] = true
		}
	case "zipfian":
		data, buckets, present = generateZipfianData(rng, cfg)
	case "nofilter":
		data = generateNoFilterData(rng, cfg)
		buckets = nil
		present = nil
	default:
		return fmt.Errorf("unknown distribution: %s", cfg.Distribution)
	}

	const batchSize = 1000
	pks := make([]model.ID, cfg.NumVecs)

	for start := 0; start < cfg.NumVecs; start += batchSize {
		end := min(start+batchSize, cfg.NumVecs)
		batchVecs := data[start:end]

		var batchMds []metadata.Document
		if cfg.Distribution != "nofilter" {
			batchMds = make([]metadata.Document, end-start)
			for i := range batchMds {
				if present[start+i] {
					batchMds[i] = metadata.Document{"bucket": metadata.Int(buckets[start+i])}
				} else {
					batchMds[i] = metadata.Document{}
				}
			}
		}

		ids, err := db.BatchInsert(ctx, batchVecs, batchMds, nil)
		if err != nil {
			return fmt.Errorf("batch insert: %w", err)
		}
		copy(pks[start:end], ids)
	}

	// Commit to persist data to disk (required for fixture to be reusable)
	if err := db.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	const numQueries = 100
	queries := make([][]float32, numQueries)
	queryRng := testutil.NewRNG(cfg.Seed + 1000)
	for i := range queries {
		q := make([]float32, cfg.Dim)
		queryRng.FillUniform(q)
		queries[i] = q
	}

	selectivities := []float64{0.01, 0.05, 0.10, 0.30, 0.50, 0.90}
	groundTruth := make(map[string][][]model.ID)

	if cfg.Distribution != "nofilter" {
		for _, sel := range selectivities {
			key := fmt.Sprintf("%.2f", sel)
			threshold := int64(float64(cfg.BucketCount) * sel)
			if threshold < 1 {
				threshold = 1
			}

			filteredData := make([][]float32, 0)
			filteredPKs := make([]model.ID, 0)
			for i := range cfg.NumVecs {
				if present[i] && buckets[i] < threshold {
					filteredData = append(filteredData, data[i])
					filteredPKs = append(filteredPKs, pks[i])
				}
			}

			truth := make([][]model.ID, numQueries)
			for qi, q := range queries {
				truth[qi] = exactTopK_L2_WithIDs(filteredData, filteredPKs, q, 10)
			}
			groundTruth[key] = truth
		}
	} else {
		truth := make([][]model.ID, numQueries)
		for qi, q := range queries {
			truth[qi] = exactTopK_L2_WithIDs(data, pks, q, 10)
		}
		groundTruth["1.00"] = truth
	}

	if err := saveFixtureMeta(cfg, dir); err != nil {
		return fmt.Errorf("save meta: %w", err)
	}
	if err := saveQueries(queries, dir); err != nil {
		return fmt.Errorf("save queries: %w", err)
	}
	if err := saveGroundTruth(groundTruth, dir); err != nil {
		return fmt.Errorf("save ground truth: %w", err)
	}
	if cfg.Distribution != "nofilter" {
		if err := saveBuckets(buckets, present, pks, dir); err != nil {
			return fmt.Errorf("save buckets: %w", err)
		}
	}

	return nil
}

func generateUniformData(rng *testutil.RNG, cfg FixtureConfig) ([][]float32, []int64) {
	data := make([][]float32, cfg.NumVecs)
	buckets := make([]int64, cfg.NumVecs)
	for i := range cfg.NumVecs {
		vec := make([]float32, cfg.Dim)
		rng.FillUniform(vec)
		data[i] = vec
		buckets[i] = int64(i) % int64(cfg.BucketCount)
	}
	return data, buckets
}

func generateZipfianData(rng *testutil.RNG, cfg FixtureConfig) ([][]float32, []int64, []bool) {
	buckets := rng.ZipfBuckets(cfg.NumVecs, cfg.BucketCount, cfg.ZipfSkew)
	present := rng.SparseMetadata(cfg.NumVecs, cfg.MissingRate)
	data := rng.ClusteredVectorsWithBuckets(cfg.NumVecs, cfg.Dim, cfg.BucketCount, buckets, cfg.VectorNoise)
	return data, buckets, present
}

func generateNoFilterData(rng *testutil.RNG, cfg FixtureConfig) [][]float32 {
	data := make([][]float32, cfg.NumVecs)
	for i := range cfg.NumVecs {
		vec := make([]float32, cfg.Dim)
		rng.FillUniform(vec)
		data[i] = vec
	}
	return data
}

// ============================================================================
// Persistence
// ============================================================================

func saveFixtureMeta(cfg FixtureConfig, dir string) error {
	path := filepath.Join(dir, fixtureMetaFile)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = fmt.Fprintf(f, "name=%s\ndim=%d\nnum_vecs=%d\nbucket_count=%d\ndistribution=%s\nzipf_skew=%.2f\nvector_noise=%.2f\nmissing_rate=%.2f\nseed=%d\n",
		cfg.Name, cfg.Dim, cfg.NumVecs, cfg.BucketCount, cfg.Distribution,
		cfg.ZipfSkew, cfg.VectorNoise, cfg.MissingRate, cfg.Seed)
	return err
}

func saveQueries(queries [][]float32, dir string) error {
	path := filepath.Join(dir, queryFile)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	numQueries := uint32(len(queries))
	dim := uint32(0)
	if len(queries) > 0 {
		dim = uint32(len(queries[0]))
	}

	if err := binary.Write(f, binary.LittleEndian, numQueries); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, dim); err != nil {
		return err
	}
	for _, q := range queries {
		if err := binary.Write(f, binary.LittleEndian, q); err != nil {
			return err
		}
	}
	return nil
}

func saveGroundTruth(truth map[string][][]model.ID, dir string) error {
	path := filepath.Join(dir, groundTruthFile)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	numSel := uint32(len(truth))
	if err := binary.Write(f, binary.LittleEndian, numSel); err != nil {
		return err
	}

	for sel, queries := range truth {
		selBytes := []byte(sel)
		if err := binary.Write(f, binary.LittleEndian, uint8(len(selBytes))); err != nil {
			return err
		}
		if _, err := f.Write(selBytes); err != nil {
			return err
		}

		numQueries := uint32(len(queries))
		if err := binary.Write(f, binary.LittleEndian, numQueries); err != nil {
			return err
		}

		for _, ids := range queries {
			k := uint32(len(ids))
			if err := binary.Write(f, binary.LittleEndian, k); err != nil {
				return err
			}
			for _, id := range ids {
				if err := binary.Write(f, binary.LittleEndian, uint64(id)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func saveBuckets(buckets []int64, present []bool, pks []model.ID, dir string) error {
	path := filepath.Join(dir, "buckets.bin")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	n := uint32(len(buckets))
	if err := binary.Write(f, binary.LittleEndian, n); err != nil {
		return err
	}

	for i := range buckets {
		if err := binary.Write(f, binary.LittleEndian, buckets[i]); err != nil {
			return err
		}
		presentByte := uint8(0)
		if present[i] {
			presentByte = 1
		}
		if err := binary.Write(f, binary.LittleEndian, presentByte); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(pks[i])); err != nil {
			return err
		}
	}
	return nil
}

// ============================================================================
// Loading
// ============================================================================

// LoadFixtureData loads pre-computed queries and ground truth.
func LoadFixtureData(name string) (*FixtureData, error) {
	fixtureCacheMu.RLock()
	if data, ok := fixtureCache[name]; ok {
		fixtureCacheMu.RUnlock()
		return data, nil
	}
	fixtureCacheMu.RUnlock()

	dir := FixtureDir(name)

	cfg, err := loadFixtureMeta(dir)
	if err != nil {
		return nil, fmt.Errorf("load meta: %w", err)
	}

	queries, err := loadQueries(dir)
	if err != nil {
		return nil, fmt.Errorf("load queries: %w", err)
	}

	groundTruth, err := loadGroundTruth(dir)
	if err != nil {
		return nil, fmt.Errorf("load ground truth: %w", err)
	}

	var buckets []int64
	var present []bool
	var pks []model.ID
	if cfg.Distribution != "nofilter" {
		buckets, present, pks, err = loadBuckets(dir)
		if err != nil {
			return nil, fmt.Errorf("load buckets: %w", err)
		}
	}

	data := &FixtureData{
		Config:      cfg,
		Queries:     queries,
		GroundTruth: groundTruth,
		Buckets:     buckets,
		Present:     present,
		PKs:         pks,
	}

	fixtureCacheMu.Lock()
	fixtureCache[name] = data
	fixtureCacheMu.Unlock()

	return data, nil
}

func loadFixtureMeta(dir string) (FixtureConfig, error) {
	path := filepath.Join(dir, fixtureMetaFile)
	content, err := os.ReadFile(path)
	if err != nil {
		return FixtureConfig{}, err
	}

	var cfg FixtureConfig
	_, err = fmt.Sscanf(string(content),
		"name=%s\ndim=%d\nnum_vecs=%d\nbucket_count=%d\ndistribution=%s\nzipf_skew=%f\nvector_noise=%f\nmissing_rate=%f\nseed=%d\n",
		&cfg.Name, &cfg.Dim, &cfg.NumVecs, &cfg.BucketCount, &cfg.Distribution,
		&cfg.ZipfSkew, &cfg.VectorNoise, &cfg.MissingRate, &cfg.Seed)
	return cfg, err
}

func loadQueries(dir string) ([][]float32, error) {
	path := filepath.Join(dir, queryFile)
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var numQueries, dim uint32
	if err := binary.Read(f, binary.LittleEndian, &numQueries); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
		return nil, err
	}

	queries := make([][]float32, numQueries)
	for i := range queries {
		q := make([]float32, dim)
		if err := binary.Read(f, binary.LittleEndian, q); err != nil {
			return nil, err
		}
		queries[i] = q
	}
	return queries, nil
}

func loadGroundTruth(dir string) (map[string][][]model.ID, error) {
	path := filepath.Join(dir, groundTruthFile)
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var numSel uint32
	if err := binary.Read(f, binary.LittleEndian, &numSel); err != nil {
		return nil, err
	}

	truth := make(map[string][][]model.ID)
	for range numSel {
		var selLen uint8
		if err := binary.Read(f, binary.LittleEndian, &selLen); err != nil {
			return nil, err
		}
		selBytes := make([]byte, selLen)
		if _, err := f.Read(selBytes); err != nil {
			return nil, err
		}
		sel := string(selBytes)

		var numQueries uint32
		if err := binary.Read(f, binary.LittleEndian, &numQueries); err != nil {
			return nil, err
		}

		queries := make([][]model.ID, numQueries)
		for qi := range queries {
			var k uint32
			if err := binary.Read(f, binary.LittleEndian, &k); err != nil {
				return nil, err
			}
			ids := make([]model.ID, k)
			for ki := range ids {
				var id uint64
				if err := binary.Read(f, binary.LittleEndian, &id); err != nil {
					return nil, err
				}
				ids[ki] = model.ID(id)
			}
			queries[qi] = ids
		}
		truth[sel] = queries
	}
	return truth, nil
}

func loadBuckets(dir string) ([]int64, []bool, []model.ID, error) {
	path := filepath.Join(dir, "buckets.bin")
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	var n uint32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, nil, nil, err
	}

	buckets := make([]int64, n)
	present := make([]bool, n)
	pks := make([]model.ID, n)

	for i := range n {
		if err := binary.Read(f, binary.LittleEndian, &buckets[i]); err != nil {
			return nil, nil, nil, err
		}
		var presentByte uint8
		if err := binary.Read(f, binary.LittleEndian, &presentByte); err != nil {
			return nil, nil, nil, err
		}
		present[i] = presentByte == 1
		var pk uint64
		if err := binary.Read(f, binary.LittleEndian, &pk); err != nil {
			return nil, nil, nil, err
		}
		pks[i] = model.ID(pk)
	}
	return buckets, present, pks, nil
}

// ============================================================================
// Fixture DB Opening
// ============================================================================

// OpenFixture opens a pre-built fixture database.
func OpenFixture(ctx context.Context, name string) (*vecgo.DB, error) {
	if !FixtureExists(name) {
		return nil, fmt.Errorf("fixture %q not found", name)
	}

	dir := FixtureDir(name)
	return vecgo.Open(ctx, vecgo.Local(dir),
		vecgo.WithCompactionThreshold(1<<40),
		vecgo.WithFlushConfig(vecgo.FlushConfig{MaxMemTableSize: 64 << 20}),
		vecgo.WithMemoryLimit(0),
	)
}

// ============================================================================
// Filter Helpers
// ============================================================================

// CreateSelectivityFilter creates a filter for a given selectivity level.
func CreateSelectivityFilter(sel float64, bucketCount int) *metadata.FilterSet {
	threshold := int64(float64(bucketCount) * sel)
	if threshold < 1 {
		threshold = 1
	}
	return metadata.NewFilterSet(metadata.Filter{
		Key:      "bucket",
		Operator: metadata.OpLessThan,
		Value:    metadata.Int(threshold),
	})
}

// SelectivityKey returns the ground truth map key for a selectivity.
func SelectivityKey(sel float64) string {
	return fmt.Sprintf("%.2f", sel)
}
