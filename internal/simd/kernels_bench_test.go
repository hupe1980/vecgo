package simd

import (
	"math"
	"math/rand"
	"testing"
)

// Benchmarks in this package are meant to be run twice to compare:
// - default build: asm enabled (SIMD dispatch when available)
// - generic build: `-tags noasm` (forces pure-Go implementations)
//
// Examples:
//   go test ./internal/simd -run '^$' -bench . -benchmem
//   go test ./internal/simd -run '^$' -bench . -benchmem -tags noasm
//
// On arm64 you can compare NEON vs noasm locally.
// On amd64, run the same commands on AVX2/AVX-512 hardware to compare.

func benchRand() *rand.Rand { return rand.New(rand.NewSource(1)) }

func randFloats(r *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = r.Float32()*2 - 1
	}
	return out
}

func randBytes(r *rand.Rand, n int) []byte {
	out := make([]byte, n)
	_, _ = r.Read(out)
	return out
}

func randInt8(r *rand.Rand, n int) []int8 {
	out := make([]int8, n)
	for i := range out {
		out[i] = int8(r.Intn(256) - 128)
	}
	return out
}

func randU16(r *rand.Rand, n int) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		out[i] = uint16(r.Uint32())
	}
	return out
}

func BenchmarkDot_Dims(b *testing.B) {
	r := benchRand()
	for _, dim := range []int{128, 256, 768, 1536} {
		b.Run("dim="+itoa(dim), func(b *testing.B) {
			a := randFloats(r, dim)
			c := randFloats(r, dim)
			b.SetBytes(int64(dim * 4 * 2))
			b.ResetTimer()
			var sink float32
			for b.Loop() {
				sink = Dot(a, c)
			}
			_ = sink
		})
	}
}

func BenchmarkSquaredL2_Dims(b *testing.B) {
	r := benchRand()
	for _, dim := range []int{128, 256, 768, 1536} {
		b.Run("dim="+itoa(dim), func(b *testing.B) {
			a := randFloats(r, dim)
			c := randFloats(r, dim)
			b.SetBytes(int64(dim * 4 * 2))
			b.ResetTimer()
			var sink float32
			for b.Loop() {
				sink = SquaredL2(a, c)
			}
			_ = sink
		})
	}
}

func BenchmarkDotBatch(b *testing.B) {
	r := benchRand()
	const dim = 256
	const n = 256
	query := randFloats(r, dim)
	targets := randFloats(r, n*dim)
	out := make([]float32, n)
	b.SetBytes(int64(n * dim * 4))
	b.ResetTimer()
	for b.Loop() {
		DotBatch(query, targets, dim, out)
	}
}

func BenchmarkSquaredL2Batch(b *testing.B) {
	r := benchRand()
	const dim = 256
	const n = 256
	query := randFloats(r, dim)
	targets := randFloats(r, n*dim)
	out := make([]float32, n)
	b.SetBytes(int64(n * dim * 4))
	b.ResetTimer()
	for b.Loop() {
		SquaredL2Batch(query, targets, dim, out)
	}
}

func BenchmarkScaleInPlace(b *testing.B) {
	r := benchRand()
	const n = 1 << 20
	a := randFloats(r, n)
	// Slightly > 1, but tiny, so values won't explode for typical b.N.
	scalar := math.Float32frombits(0x3f800001)
	b.SetBytes(int64(n * 4))
	b.ResetTimer()
	for b.Loop() {
		ScaleInPlace(a, scalar)
	}
}

func BenchmarkPqAdcLookup_M(b *testing.B) {
	r := benchRand()
	for _, m := range []int{8, 16, 32, 64} {
		b.Run("m="+itoa(m), func(b *testing.B) {
			table := randFloats(r, m*256)
			codes := randBytes(r, m)
			b.SetBytes(int64(m * 256 * 4))
			b.ResetTimer()
			var sink float32
			for b.Loop() {
				sink = PqAdcLookup(table, codes, m)
			}
			_ = sink
		})
	}
}

func BenchmarkF16ToF32(b *testing.B) {
	r := benchRand()
	const n = 1 << 20
	in := randU16(r, n)
	out := make([]float32, n)
	b.SetBytes(int64(n * 2))
	b.ResetTimer()
	for b.Loop() {
		F16ToF32(in, out)
	}
}

func BenchmarkSq8L2Batch(b *testing.B) {
	r := benchRand()
	const dim = 128
	const n = 256
	query := randFloats(r, dim)
	codes := randInt8(r, n*dim)
	scales := randFloats(r, n)
	biases := randFloats(r, n)
	out := make([]float32, n)
	b.SetBytes(int64(n * dim))
	b.ResetTimer()
	for b.Loop() {
		Sq8L2Batch(query, codes, scales, biases, dim, out)
	}
}

func BenchmarkSq8uL2BatchPerDimension(b *testing.B) {
	r := benchRand()
	const dim = 128
	const n = 256
	query := randFloats(r, dim)
	codes := randBytes(r, n*dim)
	mins := randFloats(r, dim)
	invScales := randFloats(r, dim)
	out := make([]float32, n)
	b.SetBytes(int64(n * dim))
	b.ResetTimer()
	for b.Loop() {
		Sq8uL2BatchPerDimension(query, codes, mins, invScales, dim, out)
	}
}

func BenchmarkPopcount_Sizes(b *testing.B) {
	r := benchRand()
	for _, n := range []int{128, 768, 4096, 1 << 20} {
		b.Run("n="+itoa(n), func(b *testing.B) {
			in := randBytes(r, n)
			b.SetBytes(int64(n))
			b.ResetTimer()
			var sink int64
			for b.Loop() {
				sink = Popcount(in)
			}
			_ = sink
		})
	}
}

func BenchmarkHamming_Sizes(b *testing.B) {
	r := benchRand()
	for _, n := range []int{128, 768, 4096, 1 << 20} {
		b.Run("n="+itoa(n), func(b *testing.B) {
			a := randBytes(r, n)
			c := randBytes(r, n)
			b.SetBytes(int64(n))
			b.ResetTimer()
			var sink int64
			for b.Loop() {
				sink = Hamming(a, c)
			}
			_ = sink
		})
	}
}

func BenchmarkInt8PQ_SquaredL2Dequantized(b *testing.B) {
	r := benchRand()
	for _, subdim := range []int{8, 16, 24, 32} {
		b.Run("subdim="+itoa(subdim), func(b *testing.B) {
			q := randFloats(r, subdim)
			code := randInt8(r, subdim)
			scale := float32(0.01)
			offset := float32(-0.5)
			b.SetBytes(int64(subdim * 4))
			b.ResetTimer()
			var sink float32
			for b.Loop() {
				sink = SquaredL2Int8Dequantized(q, code, scale, offset)
			}
			_ = sink
		})
	}
}

func BenchmarkInt8PQ_BuildDistanceTable(b *testing.B) {
	r := benchRand()
	for _, subdim := range []int{8, 16, 24, 32} {
		b.Run("subdim="+itoa(subdim), func(b *testing.B) {
			q := randFloats(r, subdim)
			codebook := randInt8(r, 256*subdim)
			out := make([]float32, 256)
			scale := float32(0.01)
			offset := float32(-0.5)
			b.SetBytes(int64(256 * subdim))
			b.ResetTimer()
			for b.Loop() {
				BuildDistanceTableInt8(q, codebook, subdim, scale, offset, out)
			}
		})
	}
}

func BenchmarkInt8PQ_FindNearestCentroid(b *testing.B) {
	r := benchRand()
	for _, subdim := range []int{8, 16, 24, 32} {
		b.Run("subdim="+itoa(subdim), func(b *testing.B) {
			q := randFloats(r, subdim)
			codebook := randInt8(r, 256*subdim)
			scale := float32(0.01)
			offset := float32(-0.5)
			b.SetBytes(int64(256 * subdim))
			b.ResetTimer()
			var sink int
			for b.Loop() {
				sink = FindNearestCentroidInt8(q, codebook, subdim, scale, offset)
			}
			_ = sink
		})
	}
}

func BenchmarkSqrt(b *testing.B) {
	x := float32(123.456)
	b.ResetTimer()
	var sink float32
	for b.Loop() {
		sink = Sqrt(x)
		x += 0.0001
	}
	_ = sink
}

// itoa is a tiny, allocation-free int-to-decimal helper for benchmark names.
func itoa(x int) string {
	if x == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	n := x
	if n < 0 {
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if x < 0 {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
