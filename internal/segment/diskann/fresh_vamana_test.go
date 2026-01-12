package diskann

import (
	"context"
	"math/rand"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/model"
)

func TestFreshVamanaBasic(t *testing.T) {
	ctx := context.Background()
	dim := 128
	n := 1000
	k := 10

	fv, err := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	if err != nil {
		t.Fatalf("NewFreshVamana failed: %v", err)
	}
	defer fv.Close()

	// Generate random vectors
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
	}

	// Insert all vectors
	for i := 0; i < n; i++ {
		_, err := fv.Insert(ctx, model.ID(i+1), vectors[i], nil)
		if err != nil {
			t.Fatalf("Insert failed at %d: %v", i, err)
		}
	}

	if fv.Len() != n {
		t.Errorf("Expected Len=%d, got %d", n, fv.Len())
	}

	// Search with query = first vector
	results, err := fv.Search(ctx, vectors[0], k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}

	// First result should be vector 0 itself (distance ≈ 0)
	if results[0].ID != model.ID(1) {
		t.Errorf("Expected first result to be ID=1, got ID=%d, distance=%f", results[0].ID, results[0].Distance)
	}

	// Verify results are sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("Results not sorted: results[%d].Distance=%f < results[%d].Distance=%f",
				i, results[i].Distance, i-1, results[i-1].Distance)
		}
	}
}

func TestFreshVamanaDelete(t *testing.T) {
	ctx := context.Background()
	dim := 32
	n := 100

	fv, err := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	if err != nil {
		t.Fatalf("NewFreshVamana failed: %v", err)
	}
	defer fv.Close()

	// Insert vectors
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
		fv.Insert(ctx, model.ID(i+1), vec, nil)
	}

	// Delete half the vectors
	for i := 0; i < n/2; i++ {
		err := fv.Delete(model.ID(i + 1))
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
	}

	if fv.Len() != n/2 {
		t.Errorf("Expected Len=%d after deletes, got %d", n/2, fv.Len())
	}

	// Search should not return deleted IDs
	results, err := fv.Search(ctx, vectors[0], 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	for _, r := range results {
		if r.ID <= model.ID(n/2) {
			t.Errorf("Search returned deleted ID=%d", r.ID)
		}
	}
}

func TestFreshVamanaConcurrentInsert(t *testing.T) {
	ctx := context.Background()
	dim := 64
	n := 500
	workers := 8

	fv, err := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	if err != nil {
		t.Fatalf("NewFreshVamana failed: %v", err)
	}
	defer fv.Close()

	// Generate vectors
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
	}

	// Concurrent insert
	var wg sync.WaitGroup
	perWorker := n / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := w * perWorker
		end := start + perWorker
		if w == workers-1 {
			end = n
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				_, err := fv.Insert(ctx, model.ID(i+1), vectors[i], nil)
				if err != nil {
					t.Errorf("Insert failed at %d: %v", i, err)
				}
			}
		}(start, end)
	}

	wg.Wait()

	if fv.Len() != n {
		t.Errorf("Expected Len=%d, got %d", n, fv.Len())
	}
}

func TestFreshVamanaConcurrentSearchInsert(t *testing.T) {
	ctx := context.Background()
	dim := 64
	n := 200
	k := 10

	fv, err := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	if err != nil {
		t.Fatalf("NewFreshVamana failed: %v", err)
	}
	defer fv.Close()

	// Pre-insert some vectors
	preInsert := 100
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
	}

	for i := 0; i < preInsert; i++ {
		fv.Insert(ctx, model.ID(i+1), vectors[i], nil)
	}

	// Run concurrent searches and inserts
	var wg sync.WaitGroup

	// Inserters
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := preInsert; i < n; i++ {
			_, err := fv.Insert(ctx, model.ID(i+1), vectors[i], nil)
			if err != nil {
				t.Errorf("Insert failed: %v", err)
			}
		}
	}()

	// Searchers
	for s := 0; s < 4; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				idx := rand.Intn(preInsert)
				results, err := fv.Search(ctx, vectors[idx], k)
				if err != nil {
					t.Errorf("Search failed: %v", err)
				}
				_ = results
			}
		}()
	}

	wg.Wait()
}

func TestFreshVamanaRecall(t *testing.T) {
	ctx := context.Background()
	dim := 128
	n := 1000
	k := 10
	queries := 50

	fv, err := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	if err != nil {
		t.Fatalf("NewFreshVamana failed: %v", err)
	}
	defer fv.Close()

	// Generate and insert vectors
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
		fv.Insert(ctx, model.ID(i+1), vec, nil)
	}

	// Brute force search for ground truth
	bruteSearch := func(query []float32, k int) []model.ID {
		type dist struct {
			id   model.ID
			dist float32
		}
		dists := make([]dist, n)
		distFunc, _ := distance.Provider(distance.MetricL2)
		for i := 0; i < n; i++ {
			dists[i] = dist{id: model.ID(i + 1), dist: distFunc(query, vectors[i])}
		}
		// Sort by distance
		for i := 0; i < len(dists)-1; i++ {
			for j := i + 1; j < len(dists); j++ {
				if dists[j].dist < dists[i].dist {
					dists[i], dists[j] = dists[j], dists[i]
				}
			}
		}
		result := make([]model.ID, k)
		for i := 0; i < k; i++ {
			result[i] = dists[i].id
		}
		return result
	}

	// Measure recall
	totalRecall := 0.0
	for q := 0; q < queries; q++ {
		query := vectors[rand.Intn(n)]

		groundTruth := bruteSearch(query, k)
		results, err := fv.Search(ctx, query, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Count matches
		matches := 0
		resultSet := make(map[model.ID]bool)
		for _, r := range results {
			resultSet[r.ID] = true
		}
		for _, gt := range groundTruth {
			if resultSet[gt] {
				matches++
			}
		}
		totalRecall += float64(matches) / float64(k)
	}

	avgRecall := totalRecall / float64(queries)
	t.Logf("Average recall@%d: %.2f%%", k, avgRecall*100)

	// Expect reasonable recall (>80%)
	if avgRecall < 0.80 {
		t.Errorf("Recall too low: %.2f%% (expected >80%%)", avgRecall*100)
	}
}

func BenchmarkFreshVamanaInsert(b *testing.B) {
	ctx := context.Background()
	dim := 128

	fv, _ := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	defer fv.Close()

	// Pre-generate vectors
	vectors := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fv.Insert(ctx, model.ID(i+1), vectors[i], nil)
	}
}

func BenchmarkFreshVamanaSearch(b *testing.B) {
	ctx := context.Background()
	dim := 128
	n := 10000
	k := 10

	fv, _ := NewFreshVamana(dim, distance.MetricL2, DefaultFreshVamanaOptions())
	defer fv.Close()

	// Insert vectors
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		vectors[i] = vec
		fv.Insert(ctx, model.ID(i+1), vec, nil)
	}

	// Generate queries
	queries := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		queries[i] = vec
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fv.Search(ctx, queries[i], k)
	}
}
