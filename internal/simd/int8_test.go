package simd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInt8PQHelpers_MatchGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	close := func(expected, got float32, msg string, args ...any) {
		diff := math.Abs(float64(expected - got))
		// Allow tiny absolute error + small relative error (FMA/reduction order).
		tol := math.Max(1e-2, 1e-6*math.Abs(float64(expected)))
		require.LessOrEqualf(t, diff, tol, msg, args...)
	}

	for _, subdim := range []int{1, 2, 3, 4, 7, 8, 9, 15, 16, 31, 32} {
		query := make([]float32, subdim)
		for i := range query {
			query[i] = (r.Float32()*2 - 1) * 3
		}

		codebook := make([]int8, subdim*256)
		for i := range codebook {
			codebook[i] = int8(r.Intn(256) - 128)
		}

		scale := r.Float32()*2 + 0.01
		offset := (r.Float32()*2 - 1) * 2

		// BuildDistanceTable
		expectedTable := make([]float32, 256)
		gotTable := make([]float32, 256)
		buildDistanceTableInt8Generic(query, codebook, subdim, scale, offset, expectedTable)
		BuildDistanceTableInt8(query, codebook, subdim, scale, offset, gotTable)
		for i := 0; i < 256; i++ {
			close(expectedTable[i], gotTable[i], "subdim=%d centroid=%d", subdim, i)
		}

		// SquaredL2Int8Dequantized (pick a centroid)
		centroid := r.Intn(256)
		code := codebook[centroid*subdim : (centroid+1)*subdim]
		expectedDist := squaredL2Int8DequantizedGeneric(query, code, scale, offset)
		gotDist := SquaredL2Int8Dequantized(query, code, scale, offset)
		close(expectedDist, gotDist, "subdim=%d", subdim)

		// FindNearestCentroidInt8
		expectedIdx := findNearestCentroidInt8Generic(query, codebook, subdim, scale, offset)
		gotIdx := FindNearestCentroidInt8(query, codebook, subdim, scale, offset)
		require.Equalf(t, expectedIdx, gotIdx, "subdim=%d", subdim)
	}
}
