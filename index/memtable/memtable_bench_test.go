package memtable

import (
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkMemTableSearch(b *testing.B) {
	dim := 128
	size := 10000
	k := 10

	m := New(dim, distance.SquaredL2)
	rng := testutil.NewRNG(42)

	for i := 0; i < size; i++ {
		vec := rng.UnitVector(dim)
		m.Insert(uint64(i), vec)
	}

	query := rng.UnitVector(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Search(query, k, nil)
	}
}
