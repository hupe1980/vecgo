// Package math32 provides SIMD-optimized float32 vector operations.
// This is an internal package - external users should use the distance package.
package math32

var (
	dotImpl       = dotGeneric
	squaredL2Impl = squaredL2Generic
	scaleImpl     = scaleGeneric
	pqAdcImpl     = pqAdcLookupGeneric
)

// Dot calculates the dot product of two vectors.
// Public for use by the distance package.
//
// SAFETY: This function assumes len(a) == len(b).
// It does NOT perform bounds checks for performance reasons.
// Callers MUST ensure lengths match to avoid buffer over-reads (especially with SIMD).
func Dot(a, b []float32) float32 {
	return dotImpl(a, b)
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
//
// SAFETY: This function assumes len(a) == len(b).
// It does NOT perform bounds checks for performance reasons.
// Callers MUST ensure lengths match to avoid buffer over-reads (especially with SIMD).
func SquaredL2(a, b []float32) float32 {
	return squaredL2Impl(a, b)
}

// ScaleInPlace multiplies all elements of a by scalar.
//
// This is primarily used by distance normalization.
func ScaleInPlace(a []float32, scalar float32) {
	scaleImpl(a, scalar)
}

// PqAdcLookup computes the sum of distances from a precomputed table.
// table: M x 256 floats (flattened)
// codes: M bytes
// m: number of subvectors
func PqAdcLookup(table []float32, codes []byte, m int) float32 {
	return pqAdcImpl(table, codes, m)
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

func pqAdcLookupGeneric(table []float32, codes []byte, m int) float32 {
	var sum float32
	for i := range m {
		sum += table[i*256+int(codes[i])]
	}
	return sum
}
