package benchmark_test

import (
	"testing"

	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/testutil"
)

func BenchmarkBinaryQuantizer_Distance(b *testing.B) {
	dim := 1536
	bq := quantization.NewBinaryQuantizer(dim)

	// Generate random vectors
	rng := testutil.NewRNG(1)
	vec1 := make([]float32, dim)
	vec2 := make([]float32, dim)
	rng.FillUniformRange(vec1, -0.5, 0.5)
	rng.FillUniformRange(vec2, -0.5, 0.5)

	// Encode
	code1 := make([]uint64, (dim+63)/64)
	code2 := make([]uint64, (dim+63)/64)
	bq.EncodeUint64Into(code1, vec1)
	bq.EncodeUint64Into(code2, vec2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantization.HammingDistance(code1, code2)
	}
}
