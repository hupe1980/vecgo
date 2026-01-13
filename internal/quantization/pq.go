// Package quantization provides advanced vector quantization for extreme compression.
package quantization

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/internal/mem"
	"github.com/hupe1980/vecgo/internal/simd"
)

// ProductQuantizer implements Product Quantization (PQ) for 8-32x compression.
// PQ splits vectors into subvectors and quantizes each independently using k-means clustering.
//
// Example: 128-dim vector with M=8 subvectors → 8 uint8 codes = 8 bytes (16x compression vs float32)
type ProductQuantizer struct {
	numSubvectors int       // M: number of subvectors
	numCentroids  int       // K: number of centroids per subspace (typically 256 for uint8)
	dimension     int       // D: original vector dimension
	subvectorDim  int       // D/M: dimensions per subvector
	codebooks     []int8    // Quantized centroids: M * K * subvectorDim
	scales        []float32 // Scale per subvector: M
	offsets       []float32 // Offset per subvector: M
	trained       bool
}

// NewProductQuantizer creates a new PQ quantizer.
// Parameters:
//   - dimension: Vector dimensionality (must be divisible by numSubvectors)
//   - numSubvectors: Number of subvectors to split into (M, typically 8, 16, or 32)
//   - numCentroids: Number of centroids per subspace (K, typically 256 for uint8 codes)
func NewProductQuantizer(dimension, numSubvectors, numCentroids int) (*ProductQuantizer, error) {
	if dimension <= 0 || numSubvectors <= 0 {
		return nil, errors.New("dimension and numSubvectors must be positive")
	}
	if dimension%numSubvectors != 0 {
		return nil, errors.New("dimension must be divisible by numSubvectors")
	}

	if numCentroids <= 0 {
		return nil, errors.New("numCentroids must be positive")
	}
	if numCentroids > 256 {
		return nil, errors.New("numCentroids must be <= 256 for uint8 encoding")
	}

	subvectorDim := dimension / numSubvectors
	size := numSubvectors * numCentroids * subvectorDim

	return &ProductQuantizer{
		numSubvectors: numSubvectors,
		numCentroids:  numCentroids,
		dimension:     dimension,
		subvectorDim:  subvectorDim,
		codebooks:     mem.AllocAlignedInt8(size),
		scales:        make([]float32, numSubvectors),
		offsets:       make([]float32, numSubvectors),
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
	var wg sync.WaitGroup
	// Limit concurrency to avoid OOM or excessive context switching if M is large
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))

	for m := 0; m < pq.numSubvectors; m++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(m int) {
			defer wg.Done()
			defer func() { <-sem }()

			start := m * pq.subvectorDim
			end := start + pq.subvectorDim

			// Run k-means to get centroids
			// We pass the full vectors and the subvector range to avoid allocating a new slice of slices
			centroids := pq.kmeans(vectors, start, end, pq.numCentroids, 20) // 20 iterations

			// Quantize centroids to int8
			minVal, maxVal := float32(math.MaxFloat32), float32(-math.MaxFloat32)
			for _, v := range centroids {
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
			}

			if maxVal == minVal {
				maxVal = minVal + 1e-6
			}

			scale := (maxVal - minVal) / 255.0
			// We map [min, max] to [-128, 127]
			// v = q * scale + offset
			// min = -128 * scale + offset => offset = min + 128 * scale
			offset := minVal + 128.0*scale

			pq.scales[m] = scale
			pq.offsets[m] = offset

			// Copy quantized centroids to flat codebooks
			baseOffset := m * pq.numCentroids * pq.subvectorDim
			for i, v := range centroids {
				// q = (v - offset) / scale
				//   = (v - (min + 128*scale)) / scale
				//   = (v - min)/scale - 128
				val := int(math.Round(float64((v - minVal) / scale)))
				if val < 0 {
					val = 0
				}
				if val > 255 {
					val = 255
				}
				q := int8(val - 128)
				pq.codebooks[baseOffset+i] = q
			}
		}(m)
	}

	wg.Wait()
	pq.trained = true
	return nil
}

// Encode quantizes a vector into PQ codes.
// Returns M uint8 codes (one per subvector).
func (pq *ProductQuantizer) Encode(vec []float32) ([]byte, error) {
	if !pq.trained {
		return nil, errors.New("ProductQuantizer not trained")
	}

	if len(vec) != pq.dimension {
		return nil, errors.New("vector dimension mismatch")
	}

	codes := make([]byte, pq.numSubvectors)

	// Quantize each subvector
	for m := 0; m < pq.numSubvectors; m++ {
		start := m * pq.subvectorDim
		end := start + pq.subvectorDim
		subvec := vec[start:end]

		// Find nearest centroid
		codebookStart := m * pq.numCentroids * pq.subvectorDim
		codebookEnd := codebookStart + pq.numCentroids*pq.subvectorDim

		scale := pq.scales[m]
		offset := pq.offsets[m]

		nearestIdx := pq.findNearestQuantizedCentroid(subvec, pq.codebooks[codebookStart:codebookEnd], scale, offset)
		codes[m] = uint8(nearestIdx)
	}

	return codes, nil
}

// findNearestQuantizedCentroid finds the index of the nearest quantized centroid.
func (pq *ProductQuantizer) findNearestQuantizedCentroid(vec []float32, codebook []int8, scale, offset float32) int {
	// Fused path: avoid materializing a float32 scratch centroid.
	return simd.FindNearestCentroidInt8(vec, codebook, pq.subvectorDim, scale, offset)
}

// Decode reconstructs an approximate vector from PQ codes.
func (pq *ProductQuantizer) Decode(codes []byte) ([]float32, error) {
	if !pq.trained {
		return nil, errors.New("ProductQuantizer not trained")
	}

	if len(codes) != pq.numSubvectors {
		return nil, errors.New("invalid code length")
	}

	reconstructed := make([]float32, pq.dimension)
	codebooks := pq.codebooks
	scales := pq.scales
	offsets := pq.offsets
	subvectorDim := pq.subvectorDim
	numCentroids := pq.numCentroids

	// Reconstruct each subvector from its centroid
	for m := 0; m < pq.numSubvectors; m++ {
		centroidIdx := int(codes[m])

		scale := scales[m]
		offset := offsets[m]

		// Calculate offsets
		codebookStart := m * numCentroids * subvectorDim
		centroidStart := codebookStart + centroidIdx*subvectorDim
		start := m * subvectorDim

		// Bounds check elimination hint for the inner loop
		// We know centroidStart + subvectorDim <= len(codebooks)
		// and start + subvectorDim <= len(reconstructed)
		// But it's hard to hint for dynamic ranges.
		// Let's try to slice the arrays to fixed ranges if possible, or just use local variables.

		// Slicing might help
		src := codebooks[centroidStart : centroidStart+subvectorDim]
		dst := reconstructed[start : start+subvectorDim]

		for i := 0; i < subvectorDim; i++ {
			dst[i] = float32(src[i])*scale + offset
		}
	}

	return reconstructed, nil
}

// ComputeAsymmetricDistance computes distance between a query vector and PQ codes.
// This is asymmetric distance computation (ADC) - query is full precision, database is quantized.
// Much faster than decoding and computing full distance.
func (pq *ProductQuantizer) ComputeAsymmetricDistance(query []float32, codes []byte) (float32, error) {
	if !pq.trained {
		return 0, errors.New("ProductQuantizer not trained")
	}

	var distance float32

	// Compute distance contribution from each subvector
	for m := 0; m < pq.numSubvectors; m++ {
		start := m * pq.subvectorDim
		end := start + pq.subvectorDim
		querySubvec := query[start:end]

		centroidIdx := int(codes[m])
		scale := pq.scales[m]
		offset := pq.offsets[m]

		codebookStart := m * pq.numCentroids * pq.subvectorDim
		centroidStart := codebookStart + centroidIdx*pq.subvectorDim
		centroidCode := pq.codebooks[centroidStart : centroidStart+pq.subvectorDim]

		// Squared L2 distance between query subvector and centroid (fused dequantize + L2)
		distance += simd.SquaredL2Int8Dequantized(querySubvec, centroidCode, scale, offset)
	}

	return distance, nil
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
func (pq *ProductQuantizer) kmeans(vectors [][]float32, startIdx, endIdx int, k, maxIters int) []float32 {
	centroids := pq.initializeCentroids(vectors, startIdx, endIdx, k)
	pq.runKMeansIterations(vectors, centroids, startIdx, endIdx, k, maxIters)
	return centroids
}

func (pq *ProductQuantizer) initializeCentroids(vectors [][]float32, startIdx, endIdx, k int) []float32 {
	dim := endIdx - startIdx
	centroids := make([]float32, k*dim)

	if len(vectors) < k {
		// Not enough data, return random vectors as centroids
		for i := 0; i < k; i++ {
			copy(centroids[i*dim:], vectors[i%len(vectors)][startIdx:endIdx])
		}
		return centroids
	}

	// Initialize centroids randomly (k-means++)
	firstIdx := rand.Intn(len(vectors))
	copy(centroids[0:dim], vectors[firstIdx][startIdx:endIdx])

	// minDistSq tracks each vector's squared distance to its nearest chosen centroid.
	minDistSq := make([]float32, len(vectors))
	var sum float32
	for i, vec := range vectors {
		d := simd.SquaredL2(vec[startIdx:endIdx], centroids[0:dim])
		minDistSq[i] = d
		sum += d
	}

	for c := 1; c < k; c++ {
		if sum == 0 {
			idx := rand.Intn(len(vectors))
			copy(centroids[c*dim:], vectors[idx][startIdx:endIdx])
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
		copy(centroids[c*dim:], vectors[chosen][startIdx:endIdx])

		// Update minDistSq incrementally (O(n) per centroid).
		sum = 0
		cStart := c * dim
		for i, vec := range vectors {
			d := simd.SquaredL2(vec[startIdx:endIdx], centroids[cStart:cStart+dim])
			if d < minDistSq[i] {
				minDistSq[i] = d
			}
			sum += minDistSq[i]
		}
	}
	return centroids
}

func (pq *ProductQuantizer) runKMeansIterations(vectors [][]float32, centroids []float32, startIdx, endIdx, k, maxIters int) {
	dim := endIdx - startIdx
	assignments := make([]int, len(vectors))
	numWorkers := runtime.GOMAXPROCS(0)

	for range maxIters {
		if !pq.assignClusters(vectors, centroids, assignments, startIdx, endIdx, numWorkers) {
			break
		}
		pq.updateCentroids(vectors, centroids, assignments, startIdx, endIdx, k, dim)
	}
}

func (pq *ProductQuantizer) assignClusters(vectors [][]float32, centroids []float32, assignments []int, startIdx, endIdx, numWorkers int) bool {
	var changedAtomic atomic.Bool
	var wg sync.WaitGroup

	chunkSize := (len(vectors) + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(vectors) {
			end = len(vectors)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localChanged := false
			for i := start; i < end; i++ {
				nearestIdx := pq.findNearestCentroid(vectors[i][startIdx:endIdx], centroids)
				if assignments[i] != nearestIdx {
					localChanged = true
					assignments[i] = nearestIdx
				}
			}
			if localChanged {
				changedAtomic.Store(true)
			}
		}(start, end)
	}
	wg.Wait()
	return changedAtomic.Load()
}

func (pq *ProductQuantizer) updateCentroids(vectors [][]float32, centroids []float32, assignments []int, startIdx, endIdx, k, dim int) {
	counts := make([]int, k)
	newCentroids := make([]float32, k*dim)

	for i, vec := range vectors {
		cluster := assignments[i]
		counts[cluster]++
		start := cluster * dim
		for j, val := range vec[startIdx:endIdx] {
			newCentroids[start+j] += val
		}
	}

	for i := range k {
		if counts[i] > 0 {
			start := i * dim
			for j := range dim {
				centroids[start+j] = newCentroids[start+j] / float32(counts[i])
			}
		} else {
			// Re-initialize empty cluster with random vector
			idx := rand.Intn(len(vectors))
			copy(centroids[i*dim:], vectors[idx][startIdx:endIdx])
		}
	}
}

// findNearestCentroid finds the index of the nearest centroid to a vector.
func (pq *ProductQuantizer) findNearestCentroid(vec []float32, centroids []float32) int {
	minDist := float32(math.MaxFloat32)
	nearestIdx := 0

	for i := 0; i < pq.numCentroids; i++ {
		start := i * pq.subvectorDim
		end := start + pq.subvectorDim
		centroid := centroids[start:end]

		dist := simd.SquaredL2(vec, centroid)
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

// Codebooks returns the PQ codebooks and quantization parameters.
// Returns flat codebooks (M * K * subvectorDim), scales (M), and offsets (M).
func (pq *ProductQuantizer) Codebooks() ([]int8, []float32, []float32) {
	return pq.codebooks, pq.scales, pq.offsets
}

// SetCodebooks sets the PQ codebooks directly (for loading from disk).
func (pq *ProductQuantizer) SetCodebooks(codebooks []int8, scales, offsets []float32) {
	pq.codebooks = codebooks
	pq.scales = scales
	pq.offsets = offsets
	pq.trained = true
}

// BuildDistanceTable precomputes distances from a query to all centroids.
// Returns a flattened table of size M * K where table[m*K + k] is the squared distance
// from query subvector m to centroid k.
// This enables fast ADC computation using SIMD.
func (pq *ProductQuantizer) BuildDistanceTable(query []float32) ([]float32, error) {
	if len(query) != pq.dimension {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", pq.dimension, len(query))
	}

	table := make([]float32, pq.numSubvectors*pq.numCentroids)

	for m := 0; m < pq.numSubvectors; m++ {
		start := m * pq.subvectorDim
		end := start + pq.subvectorDim
		querySubvec := query[start:end]

		scale := pq.scales[m]
		offset := pq.offsets[m]
		codebookStart := m * pq.numCentroids * pq.subvectorDim
		codebook := pq.codebooks[codebookStart : codebookStart+pq.numCentroids*pq.subvectorDim]
		out := table[m*pq.numCentroids : (m+1)*pq.numCentroids]

		// Fused path: avoid materializing float32 centroid scratch for each k.
		simd.BuildDistanceTableInt8(querySubvec, codebook, pq.subvectorDim, scale, offset, out)
	}

	return table, nil
}

// AdcDistance computes the approximate distance between a query (represented by the distance table)
// and a quantized vector (represented by codes).
func (pq *ProductQuantizer) AdcDistance(table []float32, codes []byte) (float32, error) {
	if len(codes) != pq.numSubvectors {
		return 0, errors.New("codes length mismatch")
	}
	return simd.PqAdcLookup(table, codes, pq.numSubvectors), nil
}
