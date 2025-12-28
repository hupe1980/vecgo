package vecgo_bench_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
)

// BenchmarkMetadataInsert benchmarks inserting vectors with metadata
func BenchmarkMetadataInsert(b *testing.B) {
	dim := 384

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
	if err != nil {
		b.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	b.ResetTimer()

	for i := 0; b.Loop(); i++ {
		_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
			Vector: randomVector(dim),
			Data:   i,
			Metadata: metadata.Metadata{
				"category": metadata.String(categories[i%len(categories)]),
				"score":    metadata.Int(int64(i % 100)),
			},
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFilterCompilation benchmarks filter compilation/evaluation
func BenchmarkFilterCompilation(b *testing.B) {
	filters := []struct {
		name   string
		filter *metadata.FilterSet
	}{
		{
			"Equal",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("technology")},
				},
			},
		},
		{
			"GreaterThan",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "score", Operator: metadata.OpGreaterThan, Value: metadata.Int(50)},
				},
			},
		},
		{
			"Combined",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("technology")},
					{Key: "score", Operator: metadata.OpGreaterThan, Value: metadata.Int(50)},
				},
			},
		},
	}

	// Create test document
	doc := metadata.Metadata{
		"category": metadata.String("technology"),
		"score":    metadata.Int(75),
	}

	for _, f := range filters {
		b.Run(f.name, func(b *testing.B) {
			for i := 0; b.Loop(); i++ {
				_ = f.filter.Matches(doc)
			}
		})
	}
}

// BenchmarkHybridSearchSelectivity benchmarks hybrid search at various selectivity levels
func BenchmarkHybridSearchSelectivity(b *testing.B) {
	dim := 384
	size := 10000

	// Selectivity levels (percentage of documents that match)
	selectivities := []float64{0.01, 0.1, 0.5} // 1%, 10%, 50%

	for _, selectivity := range selectivities {
		b.Run(formatPercent(selectivity), func(b *testing.B) {
			db, err := vecgo.Flat[int](dim).SquaredL2().Build()
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()

			// Insert data with controlled selectivity
			numCategories := int(1.0 / selectivity)
			for i := 0; i < size; i++ {
				_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
					Vector: randomVector(dim),
					Data:   i,
					Metadata: metadata.Metadata{
						"category": metadata.String(categories[i%numCategories%len(categories)]),
						"score":    metadata.Int(int64(i * 100 / size)),
					},
				})
				if err != nil {
					b.Fatal(err)
				}
			}

			query := randomVector(dim)
			filter := &metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("technology")},
				},
			}
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).
					KNN(10).
					WithMetadata(filter).
					Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMetadataOperators benchmarks different metadata operators
func BenchmarkMetadataOperators(b *testing.B) {
	dim := 384
	size := 10000

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
	if err != nil {
		b.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	for i := 0; i < size; i++ {
		_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
			Vector: randomVector(dim),
			Data:   i,
			Metadata: metadata.Metadata{
				"category": metadata.String(categories[i%len(categories)]),
				"score":    metadata.Int(int64(i % 100)),
			},
		})
		if err != nil {
			b.Fatal(err)
		}
	}

	operators := []struct {
		name   string
		filter *metadata.FilterSet
	}{
		{
			"LessThan",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "score", Operator: metadata.OpLessThan, Value: metadata.Int(50)},
				},
			},
		},
		{
			"GreaterThan",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "score", Operator: metadata.OpGreaterThan, Value: metadata.Int(50)},
				},
			},
		},
		{
			"Equal",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpEqual, Value: metadata.String("technology")},
				},
			},
		},
		{
			"NotEqual",
			&metadata.FilterSet{
				Filters: []metadata.Filter{
					{Key: "category", Operator: metadata.OpNotEqual, Value: metadata.String("technology")},
				},
			},
		},
	}

	query := randomVector(dim)

	for _, op := range operators {
		b.Run(op.name, func(b *testing.B) {
			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).
					KNN(10).
					WithMetadata(op.filter).
					Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMultipleFilters benchmarks combining multiple filters
func BenchmarkMultipleFilters(b *testing.B) {
	dim := 384
	size := 10000

	db, err := vecgo.Flat[int](dim).SquaredL2().Build()
	if err != nil {
		b.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	for i := 0; i < size; i++ {
		_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
			Vector: randomVector(dim),
			Data:   i,
			Metadata: metadata.Metadata{
				"category": metadata.String(categories[i%len(categories)]),
				"score":    metadata.Int(int64(i % 100)),
				"active":   metadata.Bool(i%2 == 0),
			},
		})
		if err != nil {
			b.Fatal(err)
		}
	}

	filterCounts := []int{1, 2, 3}
	query := randomVector(dim)

	for _, count := range filterCounts {
		b.Run(formatCount(count), func(b *testing.B) {
			filters := make([]metadata.Filter, 0, count)
			if count >= 1 {
				filters = append(filters, metadata.Filter{
					Key: "category", Operator: metadata.OpEqual, Value: metadata.String("technology"),
				})
			}
			if count >= 2 {
				filters = append(filters, metadata.Filter{
					Key: "score", Operator: metadata.OpGreaterThan, Value: metadata.Int(25),
				})
			}
			if count >= 3 {
				filters = append(filters, metadata.Filter{
					Key: "active", Operator: metadata.OpEqual, Value: metadata.Bool(true),
				})
			}

			filterSet := &metadata.FilterSet{Filters: filters}
			b.ResetTimer()

			for i := 0; b.Loop(); i++ {
				_, err := db.Search(query).
					KNN(10).
					WithMetadata(filterSet).
					Execute(ctx)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
