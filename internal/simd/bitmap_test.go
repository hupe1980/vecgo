package simd

import (
	"math/bits"
	"math/rand"
	"testing"
)

func TestAndWords(t *testing.T) {
	tests := []struct {
		name string
		dst  []uint64
		src  []uint64
		want []uint64
	}{
		{
			name: "Empty",
			dst:  []uint64{},
			src:  []uint64{},
			want: []uint64{},
		},
		{
			name: "Single word",
			dst:  []uint64{0xFF00FF00FF00FF00},
			src:  []uint64{0x0F0F0F0F0F0F0F0F},
			want: []uint64{0x0F000F000F000F00},
		},
		{
			name: "All ones AND all zeros",
			dst:  []uint64{^uint64(0), ^uint64(0)},
			src:  []uint64{0, 0},
			want: []uint64{0, 0},
		},
		{
			name: "Identity (AND with all ones)",
			dst:  []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
			src:  []uint64{^uint64(0), ^uint64(0)},
			want: []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
		},
		{
			name: "4 words (SIMD boundary)",
			dst:  []uint64{0xFF, 0xFF, 0xFF, 0xFF},
			src:  []uint64{0x0F, 0xF0, 0x55, 0xAA},
			want: []uint64{0x0F, 0xF0, 0x55, 0xAA},
		},
		{
			name: "5 words (SIMD + tail)",
			dst:  []uint64{0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
			src:  []uint64{0x0F, 0xF0, 0x55, 0xAA, 0x33},
			want: []uint64{0x0F, 0xF0, 0x55, 0xAA, 0x33},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint64, len(tt.dst))
			copy(dst, tt.dst)
			AndWords(dst, tt.src)
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("index %d: got 0x%X, want 0x%X", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestAndNotWords(t *testing.T) {
	tests := []struct {
		name string
		dst  []uint64
		src  []uint64
		want []uint64
	}{
		{
			name: "Empty",
			dst:  []uint64{},
			src:  []uint64{},
			want: []uint64{},
		},
		{
			name: "Single word",
			dst:  []uint64{0xFF00FF00FF00FF00},
			src:  []uint64{0x0F0F0F0F0F0F0F0F},
			want: []uint64{0xF000F000F000F000},
		},
		{
			name: "Clear all (ANDNOT with all ones)",
			dst:  []uint64{^uint64(0), ^uint64(0)},
			src:  []uint64{^uint64(0), ^uint64(0)},
			want: []uint64{0, 0},
		},
		{
			name: "Keep all (ANDNOT with zeros)",
			dst:  []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
			src:  []uint64{0, 0},
			want: []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
		},
		{
			name: "4 words (SIMD boundary)",
			dst:  []uint64{0xFF, 0xFF, 0xFF, 0xFF},
			src:  []uint64{0x0F, 0xF0, 0x55, 0xAA},
			want: []uint64{0xF0, 0x0F, 0xAA, 0x55},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint64, len(tt.dst))
			copy(dst, tt.dst)
			AndNotWords(dst, tt.src)
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("index %d: got 0x%X, want 0x%X", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestOrWords(t *testing.T) {
	tests := []struct {
		name string
		dst  []uint64
		src  []uint64
		want []uint64
	}{
		{
			name: "Empty",
			dst:  []uint64{},
			src:  []uint64{},
			want: []uint64{},
		},
		{
			name: "Single word",
			dst:  []uint64{0xFF00FF00FF00FF00},
			src:  []uint64{0x0F0F0F0F0F0F0F0F},
			want: []uint64{0xFF0FFF0FFF0FFF0F},
		},
		{
			name: "OR with zeros (identity)",
			dst:  []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
			src:  []uint64{0, 0},
			want: []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
		},
		{
			name: "OR with all ones",
			dst:  []uint64{0, 0},
			src:  []uint64{^uint64(0), ^uint64(0)},
			want: []uint64{^uint64(0), ^uint64(0)},
		},
		{
			name: "4 words (SIMD boundary)",
			dst:  []uint64{0x0F, 0xF0, 0x00, 0x00},
			src:  []uint64{0xF0, 0x0F, 0xFF, 0x00},
			want: []uint64{0xFF, 0xFF, 0xFF, 0x00},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint64, len(tt.dst))
			copy(dst, tt.dst)
			OrWords(dst, tt.src)
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("index %d: got 0x%X, want 0x%X", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestXorWords(t *testing.T) {
	tests := []struct {
		name string
		dst  []uint64
		src  []uint64
		want []uint64
	}{
		{
			name: "Empty",
			dst:  []uint64{},
			src:  []uint64{},
			want: []uint64{},
		},
		{
			name: "Single word",
			dst:  []uint64{0xFF00FF00FF00FF00},
			src:  []uint64{0x0F0F0F0F0F0F0F0F},
			want: []uint64{0xF00FF00FF00FF00F},
		},
		{
			name: "XOR with self (all zeros)",
			dst:  []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
			src:  []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
			want: []uint64{0, 0},
		},
		{
			name: "XOR with zeros (identity)",
			dst:  []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
			src:  []uint64{0, 0},
			want: []uint64{0x123456789ABCDEF0, 0xFEDCBA9876543210},
		},
		{
			name: "XOR with all ones (complement)",
			dst:  []uint64{0x0F0F0F0F0F0F0F0F},
			src:  []uint64{^uint64(0)},
			want: []uint64{0xF0F0F0F0F0F0F0F0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint64, len(tt.dst))
			copy(dst, tt.dst)
			XorWords(dst, tt.src)
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("index %d: got 0x%X, want 0x%X", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestPopcountWords(t *testing.T) {
	tests := []struct {
		name  string
		words []uint64
		want  int
	}{
		{
			name:  "Empty",
			words: []uint64{},
			want:  0,
		},
		{
			name:  "All zeros",
			words: []uint64{0, 0, 0, 0},
			want:  0,
		},
		{
			name:  "All ones single word",
			words: []uint64{^uint64(0)},
			want:  64,
		},
		{
			name:  "All ones multiple words",
			words: []uint64{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)},
			want:  256,
		},
		{
			name:  "Single bit",
			words: []uint64{1},
			want:  1,
		},
		{
			name:  "Alternating bits",
			words: []uint64{0x5555555555555555},
			want:  32,
		},
		{
			name:  "Mixed",
			words: []uint64{0xFF, 0x00, 0x0F, 0xF0},
			want:  8 + 0 + 4 + 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PopcountWords(tt.words)
			if got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

// Test equivalence between SIMD and generic implementations
func TestBitmapOps_EquivalenceBoundaries(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256}

	rng := rand.New(rand.NewSource(42))

	for _, size := range sizes {
		t.Run("", func(t *testing.T) {
			// Generate random data
			dst := make([]uint64, size)
			src := make([]uint64, size)
			for i := range dst {
				dst[i] = rng.Uint64()
				src[i] = rng.Uint64()
			}

			// Test AndWords
			t.Run("AndWords", func(t *testing.T) {
				dstCopy := make([]uint64, size)
				copy(dstCopy, dst)
				AndWords(dstCopy, src)
				for i := range dstCopy {
					want := dst[i] & src[i]
					if dstCopy[i] != want {
						t.Errorf("size=%d index=%d: got 0x%X, want 0x%X", size, i, dstCopy[i], want)
					}
				}
			})

			// Test AndNotWords
			t.Run("AndNotWords", func(t *testing.T) {
				dstCopy := make([]uint64, size)
				copy(dstCopy, dst)
				AndNotWords(dstCopy, src)
				for i := range dstCopy {
					want := dst[i] &^ src[i]
					if dstCopy[i] != want {
						t.Errorf("size=%d index=%d: got 0x%X, want 0x%X", size, i, dstCopy[i], want)
					}
				}
			})

			// Test OrWords
			t.Run("OrWords", func(t *testing.T) {
				dstCopy := make([]uint64, size)
				copy(dstCopy, dst)
				OrWords(dstCopy, src)
				for i := range dstCopy {
					want := dst[i] | src[i]
					if dstCopy[i] != want {
						t.Errorf("size=%d index=%d: got 0x%X, want 0x%X", size, i, dstCopy[i], want)
					}
				}
			})

			// Test XorWords
			t.Run("XorWords", func(t *testing.T) {
				dstCopy := make([]uint64, size)
				copy(dstCopy, dst)
				XorWords(dstCopy, src)
				for i := range dstCopy {
					want := dst[i] ^ src[i]
					if dstCopy[i] != want {
						t.Errorf("size=%d index=%d: got 0x%X, want 0x%X", size, i, dstCopy[i], want)
					}
				}
			})

			// Test PopcountWords
			t.Run("PopcountWords", func(t *testing.T) {
				got := PopcountWords(dst)
				want := 0
				for _, w := range dst {
					want += bits.OnesCount64(w)
				}
				if got != want {
					t.Errorf("size=%d: got %d, want %d", size, got, want)
				}
			})
		})
	}
}

// Benchmarks
func BenchmarkAndWords(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		dst := make([]uint64, size)
		src := make([]uint64, size)
		for i := range dst {
			dst[i] = uint64(i)
			src[i] = uint64(i * 2)
		}
		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				AndWords(dst, src)
			}
		})
	}
}

func BenchmarkOrWords(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		dst := make([]uint64, size)
		src := make([]uint64, size)
		for i := range dst {
			dst[i] = uint64(i)
			src[i] = uint64(i * 2)
		}
		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				OrWords(dst, src)
			}
		})
	}
}

func BenchmarkPopcountWords(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		words := make([]uint64, size)
		for i := range words {
			words[i] = uint64(i)
		}
		b.Run("", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				PopcountWords(words)
			}
		})
	}
}
