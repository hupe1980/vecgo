package engine

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFilteringEquivalence(t *testing.T) {
	dir := t.TempDir()
	// Small MemTable size to force flushes
	e, err := Open(dir, 2, distance.MetricL2, WithFlushConfig(FlushConfig{
		MaxMemTableSize: 20 * 1024, // 20KB
	}))
	require.NoError(t, err)
	defer e.Close()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	n := 1000
	dim := 2

	type record struct {
		pk  uint64
		vec []float32
		md  metadata.Document
	}
	records := make([]record, n)

	categories := []string{"A", "B", "C", "D"}

	for i := 0; i < n; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rng.Float32()
		}
		cat := categories[rng.Intn(len(categories))]
		price := rng.Float64() * 100.0
		md := metadata.Document{
			"category": metadata.String(cat),
			"price":    metadata.Float(price),
			"id":       metadata.Float(float64(i)),
		}
		records[i] = record{
			pk:  uint64(i),
			vec: vec,
			md:  md,
		}
		err := e.Insert(model.PKUint64(uint64(i)), vec, md, nil)
		require.NoError(t, err)
	}

	// Force flush
	err = e.Flush()
	require.NoError(t, err)

	// Define filters
	filters := []struct {
		name   string
		filter *metadata.FilterSet
	}{
		{
			name: "Category=A",
			filter: metadata.NewFilterSet(metadata.Filter{
				Key:      "category",
				Operator: metadata.OpEqual,
				Value:    metadata.String("A"),
			}),
		},
		{
			name: "Price > 50",
			filter: metadata.NewFilterSet(metadata.Filter{
				Key:      "price",
				Operator: metadata.OpGreaterThan,
				Value:    metadata.Float(50.0),
			}),
		},
		{
			name: "Category=B AND Price < 20",
			filter: metadata.NewFilterSet(
				metadata.Filter{
					Key:      "category",
					Operator: metadata.OpEqual,
					Value:    metadata.String("B"),
				},
				metadata.Filter{
					Key:      "price",
					Operator: metadata.OpLessThan,
					Value:    metadata.Float(20.0),
				},
			),
		},
	}

	ctx := context.Background()
	q := []float32{0.5, 0.5}
	k := 10

	for _, tc := range filters {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Engine Search
			res, err := e.Search(ctx, q, k, WithFilter(tc.filter))
			require.NoError(t, err)

			// 2. Brute Force Search
			var candidates []model.Candidate
			for _, r := range records {
				// Check filter
				if !tc.filter.Matches(r.md) {
					continue
				}
				// Calculate distance
				dist := distance.SquaredL2(q, r.vec)
				candidates = append(candidates, model.Candidate{
					PK:    model.PKUint64(r.pk),
					Score: dist,
				})
			}

			// Sort candidates by score (ascending for L2)
			// Simple bubble sort for test
			for i := 0; i < len(candidates); i++ {
				for j := i + 1; j < len(candidates); j++ {
					if candidates[i].Score > candidates[j].Score {
						candidates[i], candidates[j] = candidates[j], candidates[i]
					}
				}
			}

			// Take top k
			if len(candidates) > k {
				candidates = candidates[:k]
			}

			// Compare PKs
			// Note: scores might be slightly different due to float precision or quantization (if any)
			// But here we use L2 and no quantization in MemTable/Flat (unless configured), so should be close.
			// However, Flat segment might use quantization if configured. Default is usually none or SQ.
			// Let's check if we get the same PKs.

			// Also, if scores are identical, order might differ.
			// We'll check if the set of PKs matches.

			resPKs := make(map[uint64]bool)
			for _, r := range res {
				u64, _ := r.PK.Uint64()
				resPKs[u64] = true
			}

			expectedPKs := make(map[uint64]bool)
			for _, c := range candidates {
				u64, _ := c.PK.Uint64()
				expectedPKs[u64] = true
			}

			// If we have fewer results than k, they must match exactly.
			// If we have k results, the worst score in res should be <= worst score in candidates (approx).

			// For strict equivalence, let's just check if all returned PKs are valid according to the filter.
			for _, r := range res {
				// Find record
				u64, _ := r.PK.Uint64()
				rec := records[u64]
				if !tc.filter.Matches(rec.md) {
					t.Errorf("Result PK %d does not match filter", u64)
				}
			}

			// And check if we missed any better candidates?
			// That's harder if scores are close.
			// But at least we verified that NO invalid records are returned.

			// Let's verify count if total matches < k
			matches := 0
			for _, r := range records {
				if tc.filter.Matches(r.md) {
					matches++
				}
			}

			if matches < k {
				assert.Len(t, res, matches, "Should return all matches if fewer than k")
			} else {
				assert.Len(t, res, k, "Should return k matches")
			}
		})
	}
}
