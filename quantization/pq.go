// Package quantization provides advanced vector quantization for extreme compression.
package quantization

import (
	"errors"
	"math"
	"math/rand"

	"github.com/hupe1980/vecgo/internal/math32"
)

// ProductQuantizer implements Product Quantization (PQ) for 8-32x compression.
// PQ splits vectors into subvectors and quantizes each independently using k-means clustering.
//
// Example: 128-dim vector with M=8 subvectors → 8 uint8 codes = 8 bytes (16x compression vs float32)
type ProductQuantizer struct {
	numSubvectors int           // M: number of subvectors
	numCentroids  int           // K: number of centroids per subspace (typically 256 for uint8)
	dimension     int           // D: original vector dimension
	subvectorDim  int           // D/M: dimensions per subvector
	codebooks     [][][]float32 // M codebooks, each with K centroids of subvectorDim dimensions
	trained       bool
}

// NewProductQuantizer creates a new PQ quantizer.
// Parameters:
//   - dimension: Vector dimensionality (must be divisible by numSubvectors)
//   - numSubvectors: Number of subvectors to split into (M, typically 8, 16, or 32)
//   - numCentroids: Number of centroids per subspace (K, typically 256 for uint8 codes)
func NewProductQuantizer(dimension, numSubvectors, numCentroids int) (*ProductQuantizer, error) {
	if dimension%numSubvectors != 0 {
		return nil, errors.New("dimension must be divisible by numSubvectors")
	}

	if numCentroids > 256 {
		return nil, errors.New("numCentroids must be <= 256 for uint8 encoding")
	}

	return &ProductQuantizer{
		numSubvectors: numSubvectors,
		numCentroids:  numCentroids,
		dimension:     dimension,
		subvectorDim:  dimension / numSubvectors,
		codebooks:     make([][][]float32, numSubvectors),
		trained:       false,
	}, nil
}

// Train calibrates the PQ quantizer using k-means clustering on training vectors.
// This must be called before Encode/Decode.
func (pq *ProductQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided for training")
	}

	// Validate dimensions
	if len(vectors[0]) != pq.dimension {
		return errors.New("vector dimension mismatch")
	}

	// Train one codebook per subvector
	for m := 0; m < pq.numSubvectors; m++ {
		// Extract subvectors for this position
		subvectors := make([][]float32, len(vectors))
		for i, vec := range vectors {
			start := m * pq.subvectorDim
			end := start + pq.subvectorDim
			subvectors[i] = vec[start:end]
		}

		// Run k-means to get centroids
		centroids := pq.kmeans(subvectors, pq.numCentroids, 20) // 20 iterations
		pq.codebooks[m] = centroids
	}

	pq.trained = true
	return nil
}

// Encode quantizes a vector into PQ codes.
// Returns M uint8 codes (one per subvector).
func (pq *ProductQuantizer) Encode(vec []float32) []byte {
	if !pq.trained {
		panic("ProductQuantizer not trained")
	}

	if len(vec) != pq.dimension {
		panic("vector dimension mismatch")
	}

	codes := make([]byte, pq.numSubvectors)

	// Quantize each subvector
	for m := 0; m < pq.numSubvectors; m++ {
		start := m * pq.subvectorDim
		end := start + pq.subvectorDim
		subvec := vec[start:end]

		// Find nearest centroid
		nearestIdx := pq.findNearestCentroid(subvec, pq.codebooks[m])
		codes[m] = uint8(nearestIdx)
	}

	return codes
}

// Decode reconstructs an approximate vector from PQ codes.
func (pq *ProductQuantizer) Decode(codes []byte) []float32 {
	if !pq.trained {
		panic("ProductQuantizer not trained")
	}

	if len(codes) != pq.numSubvectors {
		panic("invalid code length")
	}

	reconstructed := make([]float32, pq.dimension)

	// Reconstruct each subvector from its centroid
	for m := 0; m < pq.numSubvectors; m++ {
		centroidIdx := int(codes[m])
		centroid := pq.codebooks[m][centroidIdx]

		// Copy centroid to reconstructed vector
		start := m * pq.subvectorDim
		copy(reconstructed[start:start+pq.subvectorDim], centroid)
	}

	return reconstructed
}

// ComputeAsymmetricDistance computes distance between a query vector and PQ codes.
// This is asymmetric distance computation (ADC) - query is full precision, database is quantized.
// Much faster than decoding and computing full distance.
func (pq *ProductQuantizer) ComputeAsymmetricDistance(query []float32, codes []byte) float32 {
	if !pq.trained {
		panic("ProductQuantizer not trained")
	}

	var distance float32

	// Compute distance contribution from each subvector
	for m := 0; m < pq.numSubvectors; m++ {
		start := m * pq.subvectorDim
		end := start + pq.subvectorDim
		querySubvec := query[start:end]

		centroidIdx := int(codes[m])
		centroid := pq.codebooks[m][centroidIdx]

		// Squared L2 distance between query subvector and centroid
		for i, val := range querySubvec {
			diff := val - centroid[i]
			distance += diff * diff
		}
	}

	return distance
}

// BytesPerVector returns the compressed size per vector in bytes.
func (pq *ProductQuantizer) BytesPerVector() int {
	return pq.numSubvectors // One uint8 per subvector
}

// CompressionRatio returns the theoretical compression ratio.
func (pq *ProductQuantizer) CompressionRatio() float64 {
	originalBytes := pq.dimension * 4 // float32 = 4 bytes
	compressedBytes := pq.numSubvectors
	return float64(originalBytes) / float64(compressedBytes)
}

// kmeans performs k-means clustering on subvectors.
func (pq *ProductQuantizer) kmeans(vectors [][]float32, k, maxIters int) [][]float32 {
	if len(vectors) < k {
		// Not enough data, return random vectors as centroids
		dim := len(vectors[0])
		centroids := make([][]float32, k)
		for i := range centroids {
			centroids[i] = make([]float32, dim)
			copy(centroids[i], vectors[i%len(vectors)])
		}
		return centroids
	}

	dim := len(vectors[0])

	// Initialize centroids randomly (k-means++)
	centroids := make([][]float32, k)
	for i := range centroids {
		centroids[i] = make([]float32, dim)
	}

	firstIdx := rand.Intn(len(vectors))
	copy(centroids[0], vectors[firstIdx])

	// minDistSq tracks each vector's squared distance to its nearest chosen centroid.
	minDistSq := make([]float32, len(vectors))
	var sum float32
	for i, vec := range vectors {
		d := math32.SquaredL2(vec, centroids[0])
		minDistSq[i] = d
		sum += d
	}

	for c := 1; c < k; c++ {
		if sum == 0 {
			idx := rand.Intn(len(vectors))
			copy(centroids[c], vectors[idx])
			continue
		}

		// Sample proportional to squared distance (already squared in minDistSq).
		target := rand.Float32() * sum
		var cumsum float32
		chosen := 0
		for i, d := range minDistSq {
			cumsum += d
			if cumsum >= target {
				chosen = i
				break
			}
		}
		copy(centroids[c], vectors[chosen])

		// Update minDistSq incrementally (O(n) per centroid).
		sum = 0
		for i, vec := range vectors {
			d := math32.SquaredL2(vec, centroids[c])
			if d < minDistSq[i] {
				minDistSq[i] = d
			}
			sum += minDistSq[i]
		}
	}

	// Run k-means iterations
	assignments := make([]int, len(vectors))
	for range maxIters {
		// Assignment step
		changed := false
		for i, vec := range vectors {
			nearestIdx := pq.findNearestCentroid(vec, centroids)
			if assignments[i] != nearestIdx {
				changed = true
				assignments[i] = nearestIdx
			}
		}

		if !changed {
			break
		}

		// Update step
		counts := make([]int, k)
		newCentroidsFlat := make([]float32, k*dim)
		newCentroids := make([][]float32, k)
		for i := range newCentroids {
			start := i * dim
			newCentroids[i] = newCentroidsFlat[start : start+dim]
		}

		for i, vec := range vectors {
			cluster := assignments[i]
			counts[cluster]++
			for j, val := range vec {
				newCentroids[cluster][j] += val
			}
		}

		for i := range centroids {
			if counts[i] > 0 {
				for j := range centroids[i] {
					centroids[i][j] = newCentroids[i][j] / float32(counts[i])
				}
			}
		}
	}

	return centroids
}

// findNearestCentroid finds the index of the nearest centroid to a vector.
func (pq *ProductQuantizer) findNearestCentroid(vec []float32, centroids [][]float32) int {
	minDist := float32(math.MaxFloat32)
	nearestIdx := 0

	for i, centroid := range centroids {
		dist := math32.SquaredL2(vec, centroid)
		if dist < minDist {
			minDist = dist
			nearestIdx = i
		}
	}

	return nearestIdx
}

// NumSubvectors returns the number of subvectors (M).
func (pq *ProductQuantizer) NumSubvectors() int {
	return pq.numSubvectors
}

// NumCentroids returns the number of centroids per subspace (K).
func (pq *ProductQuantizer) NumCentroids() int {
	return pq.numCentroids
}

// IsTrained returns whether the quantizer has been trained.
func (pq *ProductQuantizer) IsTrained() bool {
	return pq.trained
}

// Codebooks returns the PQ codebooks.
// Returns M codebooks, each with K centroids of subvectorDim dimensions.
func (pq *ProductQuantizer) Codebooks() [][][]float32 {
	return pq.codebooks
}

// SetCodebooks sets the PQ codebooks directly (for loading from disk).
// The codebooks must have shape [M][K][subvectorDim].
func (pq *ProductQuantizer) SetCodebooks(codebooks [][][]float32) {
	pq.codebooks = codebooks
	pq.trained = true
}

// BuildDistanceTable precomputes distances from a query to all centroids.
// Returns a flattened table of size M * K where table[m*K + k] is the squared distance
// from query subvector m to centroid k.
// This enables fast ADC computation using SIMD.
func (pq *ProductQuantizer) BuildDistanceTable(query []float32) []float32 {
	if len(query) != pq.dimension {
		panic("query dimension mismatch")
	}

	table := make([]float32, pq.numSubvectors*pq.numCentroids)
	for m := 0; m < pq.numSubvectors; m++ {
		start := m * pq.subvectorDim
		end := start + pq.subvectorDim
		querySubvec := query[start:end]

		for k := 0; k < pq.numCentroids; k++ {
			dist := math32.SquaredL2(querySubvec, pq.codebooks[m][k])
			table[m*pq.numCentroids+k] = dist
		}
	}

	return table
}

// AdcDistance computes the approximate distance between a query (represented by the distance table)
// and a quantized vector (represented by codes).
func (pq *ProductQuantizer) AdcDistance(table []float32, codes []byte) float32 {
	if len(codes) != pq.numSubvectors {
		panic("codes length mismatch")
	}
	return math32.PqAdcLookup(table, codes, pq.numSubvectors)
}
