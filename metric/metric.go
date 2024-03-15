package metric

import (
	"errors"

	"github.com/hupe1980/vecgo/internal/math32"
)

// Magnitude calculates the magnitude (length) of a float32 slice.
func Magnitude(v []float32) float32 {
	return math32.Sqrt(math32.Dot(v, v))
}

// CosineSimilarity calculates the cosine similarity between two float32 slices.
func CosineSimilarity(v1, v2 []float32) (float32, error) {
	// Check if the vector sizes match
	if len(v1) != len(v2) {
		return 0, errors.New("vector sizes do not match")
	}

	dotProduct := math32.Dot(v1, v2)
	magnitudeA := Magnitude(v1)
	magnitudeB := Magnitude(v2)

	// Avoid division by zero
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0, nil
	}

	return dotProduct / (magnitudeA * magnitudeB), nil
}

// SquaredL2 calculates the squared L2 distance between two float32 slices.
func SquaredL2(v1, v2 []float32) (float32, error) {
	// Check if the vector sizes match
	if len(v1) != len(v2) {
		return 0, errors.New("vector sizes do not match")
	}

	return math32.SquaredL2(v1, v2), nil
}
