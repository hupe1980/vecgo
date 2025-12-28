// Package math32 provides SIMD-optimized float32 vector operations.
// This is an internal package - external users should use the distance package.
package math32

var (
	useAVX    bool // nolint unused
	useAVX512 bool // nolint unused
	useNEON   bool // nolint unused
)

// Dot calculates the dot product of two vectors.
// Public for use by the distance package.
func Dot(a, b []float32) float32 {
	return dot(a, b)
}

func dotGeneric(a, b []float32) float32 {
	var ret float32
	for i := range a {
		ret += a[i] * b[i]
	}

	return ret
}

// SquaredL2 calculates the squared L2 distance.
// Public for use by the distance package.
func SquaredL2(a, b []float32) float32 {
	return squaredL2(a, b)
}

// ScaleInPlace multiplies all elements of a by scalar.
//
// This is primarily used by distance normalization.
func ScaleInPlace(a []float32, scalar float32) {
	scaleInPlace(a, scalar)
}

func squaredL2Generic(a, b []float32) float32 {
	var distance float32
	for i := range a {
		distance += (a[i] - b[i]) * (a[i] - b[i])
	}

	return distance
}

func scaleGeneric(a []float32, scalar float32) {
	for i := range a {
		a[i] *= scalar
	}
}
