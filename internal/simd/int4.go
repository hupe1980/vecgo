package simd

// int4L2DistanceGeneric computes squared L2 distance between query and INT4 code.
//
// INT4 codes are nibble-packed: high nibble first, low nibble second.
// Dequantization: val = (quant / 15.0) * diff[i] + minVal[i]
//
// SAFETY: Assumes len(query) == dim, len(code) == (dim+1)/2
// Caller's responsibility to ensure alignment.
func int4L2DistanceGeneric(query []float32, code []byte, minVal, diff []float32) float32 {
	var sum float32
	dim := len(query)

	for i := 0; i < dim; i += 2 {
		byteVal := code[i/2]

		// High nibble (first value)
		quant1 := (byteVal >> 4) & 0x0F
		val1 := float32(quant1)/15.0*diff[i] + minVal[i]
		d1 := val1 - query[i]
		sum += d1 * d1

		// Low nibble (second value)
		if i+1 < dim {
			quant2 := byteVal & 0x0F
			val2 := float32(quant2)/15.0*diff[i+1] + minVal[i+1]
			d2 := val2 - query[i+1]
			sum += d2 * d2
		}
	}

	return sum
}

var int4L2DistanceImpl = int4L2DistanceGeneric

// Int4L2Distance computes squared L2 distance between query and INT4 code.
//
// SAFETY: Assumes len(query) == dim, len(code) == (dim+1)/2
func Int4L2Distance(query []float32, code []byte, minVal, diff []float32) float32 {
	return int4L2DistanceImpl(query, code, minVal, diff)
}

// int4L2DistanceBatchGeneric computes L2 distances for multiple INT4 codes.
//
// codes is a flattened array of n codes, each of size (dim+1)/2.
// out must have length n.
func int4L2DistanceBatchGeneric(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	codeSize := (dim + 1) / 2
	for j := 0; j < n; j++ {
		code := codes[j*codeSize : (j+1)*codeSize]
		out[j] = int4L2DistanceGeneric(query, code, minVal, diff)
	}
}

var int4L2DistanceBatchImpl = int4L2DistanceBatchGeneric

// Int4L2DistanceBatch computes L2 distances for multiple INT4 codes.
//
// codes is a flattened array of n codes, each of size (dim+1)/2.
// out must have length n.
func Int4L2DistanceBatch(query []float32, codes []byte, dim, n int, minVal, diff []float32, out []float32) {
	int4L2DistanceBatchImpl(query, codes, dim, n, minVal, diff, out)
}

// int4L2DistancePrecomputedGeneric uses pre-computed dequantization tables.
// This is faster when the same quantizer parameters are used for many queries.
//
// lookupTable: 16 * dim floats, where lookupTable[i*16 + q] = (q/15.0)*diff[i] + min[i]
func int4L2DistancePrecomputedGeneric(query []float32, code []byte, lookupTable []float32) float32 {
	var sum float32
	dim := len(query)

	for i := 0; i < dim; i += 2 {
		byteVal := code[i/2]

		// High nibble
		quant1 := int((byteVal >> 4) & 0x0F)
		val1 := lookupTable[i*16+quant1]
		d1 := val1 - query[i]
		sum += d1 * d1

		// Low nibble
		if i+1 < dim {
			quant2 := int(byteVal & 0x0F)
			val2 := lookupTable[(i+1)*16+quant2]
			d2 := val2 - query[i+1]
			sum += d2 * d2
		}
	}

	return sum
}

var int4L2DistancePrecomputedImpl = int4L2DistancePrecomputedGeneric

// Int4L2DistancePrecomputed uses pre-computed dequantization tables.
// lookupTable: 16 * dim floats
func Int4L2DistancePrecomputed(query []float32, code []byte, lookupTable []float32) float32 {
	return int4L2DistancePrecomputedImpl(query, code, lookupTable)
}

// BuildInt4LookupTable builds a dequantization lookup table for INT4.
// Returns a table of 16 * dim floats.
func BuildInt4LookupTable(minVal, diff []float32) []float32 {
	dim := len(minVal)
	table := make([]float32, 16*dim)

	for i := range dim {
		for q := range 16 {
			table[i*16+q] = float32(q)/15.0*diff[i] + minVal[i]
		}
	}

	return table
}
