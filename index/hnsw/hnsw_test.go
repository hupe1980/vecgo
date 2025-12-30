package hnsw

import (
	"context"
	"fmt"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/assert"
)

type TestCases struct {
	VectorSize int
	VectorDim  int

	M         int
	EF        int
	Heuristic bool
	K         int

	Precision float64
}

func TestNew(t *testing.T) {
	h, err := New(func(o *Options) {
		o.Dimension = 16
		o.M = 8
		o.EF = 200
	})
	if !assert.NoError(t, err) {
		return
	}

	assert.Equal(t, 8, h.opts.M)
	assert.Equal(t, 8, h.maxConnectionsPerLayer)
	assert.Equal(t, 16, h.maxConnectionsLayer0)
	assert.Equal(t, 200, h.opts.EF)
}

func TestValidateInsertSearch(t *testing.T) {
	ctx := context.Background()
	tests := []TestCases{
		{
			VectorSize: 1000,
			VectorDim:  16,
			M:          8,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		// Non-heuristic (simple) selection has lower accuracy than heuristic
		{
			VectorSize: 1000,
			VectorDim:  16,
			M:          8,
			EF:         200,
			Heuristic:  false,
			Precision:  0.97,
			K:          10,
		},
		{
			VectorSize: 1000,
			VectorDim:  1024,
			M:          16,
			EF:         300,
			Heuristic:  true,
			Precision:  0.98,
			K:          10,
		},
		// NOTE: Larger validation cases are in hnsw_long_test.go behind the
		// `longtests` build tag.
		{
			VectorSize: 2000,
			VectorDim:  16,
			M:          16,
			EF:         128,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 2000,
			VectorDim:  32,
			M:          16,
			EF:         128,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(caseName(tc), func(t *testing.T) {
			runValidateInsertSearchCase(t, ctx, tc)
		})
	}
}

func TestHNSW_DotProduct_DistanceOrdering(t *testing.T) {
	ctx := context.Background()
	h, err := New(func(o *Options) {
		o.Dimension = 3
		o.DistanceType = index.DistanceTypeDotProduct
		o.M = 8
		o.EF = 50
	})
	if !assert.NoError(t, err) {
		return
	}

	id0, err := h.Insert(ctx, []float32{1, 0, 0})
	if !assert.NoError(t, err) {
		return
	}
	id1, err := h.Insert(ctx, []float32{2, 0, 0})
	if !assert.NoError(t, err) {
		return
	}
	id2, err := h.Insert(ctx, []float32{-1, 0, 0})
	if !assert.NoError(t, err) {
		return
	}

	query := []float32{1, 0, 0}

	brute, err := h.BruteSearch(ctx, query, 3, func(id uint64) bool { return true })
	if !assert.NoError(t, err) {
		return
	}
	if !assert.Len(t, brute, 3) {
		return
	}

	assert.Equal(t, id1, brute[0].ID)
	assert.Equal(t, id0, brute[1].ID)
	assert.Equal(t, id2, brute[2].ID)

	assert.InDelta(t, -distance.Dot(query, []float32{2, 0, 0}), brute[0].Distance, 1e-6)
	assert.InDelta(t, -distance.Dot(query, []float32{1, 0, 0}), brute[1].Distance, 1e-6)
	assert.InDelta(t, -distance.Dot(query, []float32{-1, 0, 0}), brute[2].Distance, 1e-6)

	knn, err := h.KNNSearch(ctx, query, 3, &index.SearchOptions{EFSearch: 100, Filter: func(id uint64) bool { return true }})
	if !assert.NoError(t, err) {
		return
	}
	if !assert.Len(t, knn, 3) {
		return
	}

	assert.Equal(t, id1, knn[0].ID)
	assert.Equal(t, id0, knn[1].ID)
	assert.Equal(t, id2, knn[2].ID)
}

func caseName(tc TestCases) string {
	return fmt.Sprintf(
		"Vec=%d,Dim=%d,Heuristic=%t,M=%d,Precision=%f",
		tc.VectorSize,
		tc.VectorDim,
		tc.Heuristic,
		tc.M,
		tc.Precision,
	)
}

func runValidateInsertSearchCase(t *testing.T, ctx context.Context, tc TestCases) {
	rng := testutil.NewRNG(4711)

	vecs := rng.UniformVectors(tc.VectorSize, tc.VectorDim)
	if len(vecs) != tc.VectorSize {
		t.Fatalf("unexpected vector count: got %d want %d", len(vecs), tc.VectorSize)
	}
	if tc.VectorSize > 0 && len(vecs[0]) != tc.VectorDim {
		t.Fatalf("unexpected vector dim: got %d want %d", len(vecs[0]), tc.VectorDim)
	}

	h, err := New(func(o *Options) {
		o.Dimension = tc.VectorDim
		o.M = tc.M
		o.EF = tc.EF
		o.Heuristic = tc.Heuristic
	})
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < len(vecs); i++ {
		if _, err := h.Insert(ctx, vecs[i]); err != nil {
			t.Fatalf("Insert failed at i=%d: %v", i, err)
		}
	}

	queryIndices := make([]int, 0, len(vecs))
	if len(vecs) >= 2_000 {
		const sampleQueries = 200
		step := len(vecs) / sampleQueries
		if step < 1 {
			step = 1
		}
		for i := 0; i < len(vecs) && len(queryIndices) < sampleQueries; i += step {
			queryIndices = append(queryIndices, i)
		}
	} else {
		for i := 0; i < len(vecs); i++ {
			queryIndices = append(queryIndices, i)
		}
	}

	groundResults := make(map[int][]uint64, len(queryIndices))
	for _, qi := range queryIndices {
		bestCandidatesBrute, err := h.BruteSearch(ctx, vecs[qi], tc.K, func(id uint64) bool { return true })
		if err != nil {
			t.Fatalf("BruteSearch failed: %v", err)
		}
		groundResults[qi] = make([]uint64, tc.K)
		for i2, item := range bestCandidatesBrute {
			groundResults[qi][i2] = item.ID
		}
	}

	hitSuccess := 0
	totalSearch := 0

	for _, qi := range queryIndices {
		bestCandidates, err := h.KNNSearch(ctx, vecs[qi], tc.K, &index.SearchOptions{
			EFSearch: tc.EF,
			Filter:   func(id uint64) bool { return true },
		})
		if err != nil {
			t.Fatalf("KNNSearch failed: %v", err)
		}
		if len(bestCandidates) == 0 {
			continue
		}
		for _, item := range bestCandidates {
			totalSearch++
			for k := tc.K - 1; k >= 0; k-- {
				if item.ID == groundResults[qi][k] {
					hitSuccess++
				}
			}
		}
	}

	if len(queryIndices) == 0 {
		t.Fatalf("no queries")
	}
	precision := float64(hitSuccess) / (float64(len(queryIndices)) * float64(tc.K))
	t.Logf("Precision => %f (%d/%d hits)", precision, hitSuccess, totalSearch)
	if precision < tc.Precision {
		t.Fatalf("precision too low: got %f want >= %f", precision, tc.Precision)
	}
}
