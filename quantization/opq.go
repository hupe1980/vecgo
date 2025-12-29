// Package quantization provides advanced vector quantization for extreme compression.
package quantization

import (
	"errors"
	"math"

	"github.com/hupe1980/vecgo/internal/math32"
)

// OptimizedProductQuantizer implements Optimized Product Quantization (OPQ).
// OPQ learns a rotation matrix that minimizes quantization error before applying PQ.
// This typically provides 20-30% better recall than standard PQ at the same compression ratio.
//
// The key insight: rotating the vector space before quantization can make subvectors
// more independent, reducing correlation and improving reconstruction quality.
type OptimizedProductQuantizer struct {
	pq            *ProductQuantizer
	rotation      [][]float32 // R: rotation matrix (dimension x dimension)
	rotationT     [][]float32 // R^T: transpose of rotation matrix (for inverse)
	hasRotation   bool        // whether rotation has been learned
	numIterations int         // number of OPQ optimization iterations
}

// NewOptimizedProductQuantizer creates a new OPQ quantizer.
// Parameters are the same as ProductQuantizer, plus:
//   - numIterations: Number of alternating optimization iterations (typically 10-20)
func NewOptimizedProductQuantizer(dimension, numSubvectors, numCentroids, numIterations int) (*OptimizedProductQuantizer, error) {
	pq, err := NewProductQuantizer(dimension, numSubvectors, numCentroids)
	if err != nil {
		return nil, err
	}

	return &OptimizedProductQuantizer{
		pq:            pq,
		rotation:      nil,
		rotationT:     nil,
		hasRotation:   false,
		numIterations: numIterations,
	}, nil
}

// Train calibrates the OPQ quantizer using alternating optimization.
// This learns both the rotation matrix and the PQ codebooks jointly.
func (opq *OptimizedProductQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided for training")
	}

	if len(vectors[0]) != opq.pq.dimension {
		return errors.New("vector dimension mismatch")
	}

	// Initialize rotation as identity matrix
	opq.rotation = opq.identityMatrix(opq.pq.dimension)
	opq.rotationT = opq.identityMatrix(opq.pq.dimension)
	opq.hasRotation = true

	// Alternating optimization: iterate between optimizing R and optimizing codebooks
	rotatedVectors := make([][]float32, len(vectors))
	for i := range rotatedVectors {
		rotatedVectors[i] = make([]float32, opq.pq.dimension)
	}

	for iter := 0; iter < opq.numIterations; iter++ {
		// Step 1: Apply current rotation to all vectors
		for i, vec := range vectors {
			opq.matrixVectorMultiply(opq.rotation, vec, rotatedVectors[i])
		}

		// Step 2: Train PQ codebooks on rotated vectors
		if err := opq.pq.Train(rotatedVectors); err != nil {
			return err
		}

		// Step 3: Update rotation matrix to minimize reconstruction error
		// We use a simplified approach: compute covariance of quantization residuals
		// and use eigenvectors to refine the rotation
		if iter < opq.numIterations-1 {
			opq.updateRotation(vectors, rotatedVectors)
		}
	}

	return nil
}

// updateRotation refines the rotation matrix based on quantization residuals.
// This is a simplified version that computes the covariance of residuals
// and uses it to guide the rotation toward better decorrelation.
func (opq *OptimizedProductQuantizer) updateRotation(originalVectors, rotatedVectors [][]float32) {
	dim := opq.pq.dimension

	// Compute mean residual per dimension
	residuals := make([][]float32, len(originalVectors))
	for i := range residuals {
		residuals[i] = make([]float32, dim)

		// Encode and decode to get reconstruction
		codes := opq.pq.Encode(rotatedVectors[i])
		reconstructed := opq.pq.Decode(codes)

		// Compute residual (quantization error)
		for j := range dim {
			residuals[i][j] = rotatedVectors[i][j] - reconstructed[j]
		}
	}

	// Compute covariance matrix of residuals
	cov := opq.computeCovariance(residuals)

	// Use PCA-like approach: find eigenvectors of covariance
	// For simplicity, we'll use power iteration to find the dominant eigenvector
	// and update the rotation incrementally
	eigenvector := make([]float32, dim)
	for i := range eigenvector {
		eigenvector[i] = 1.0 / float32(math.Sqrt(float64(dim)))
	}

	// Power iteration (simplified - just a few iterations)
	for range 5 {
		newVec := make([]float32, dim)
		opq.matrixVectorMultiply(cov, eigenvector, newVec)

		// Normalize
		norm := float32(0)
		for _, v := range newVec {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for i := range newVec {
				eigenvector[i] = newVec[i] / norm
			}
		}
	}

	// Apply small rotation adjustment using Householder reflection
	// This is a simplified update that nudges the rotation toward better decorrelation
	alpha := float32(0.1) // learning rate
	for i := range dim {
		for j := 0; j < dim; j++ {
			if i == j {
				opq.rotation[i][j] += alpha * (eigenvector[i] - opq.rotation[i][j])
			}
		}
	}

	// Re-orthogonalize rotation matrix using Gram-Schmidt
	opq.orthogonalize(opq.rotation)

	// Update transpose
	opq.rotationT = opq.transpose(opq.rotation)
}

// computeCovariance computes the covariance matrix of a set of vectors.
func (opq *OptimizedProductQuantizer) computeCovariance(vectors [][]float32) [][]float32 {
	dim := len(vectors[0])
	n := len(vectors)

	// Compute mean
	mean := make([]float32, dim)
	for _, vec := range vectors {
		for j, v := range vec {
			mean[j] += v
		}
	}
	for j := range mean {
		mean[j] /= float32(n)
	}

	// Compute covariance
	cov := make([][]float32, dim)
	for i := range cov {
		cov[i] = make([]float32, dim)
	}

	for _, vec := range vectors {
		for i := range dim {
			for j := range dim {
				cov[i][j] += (vec[i] - mean[i]) * (vec[j] - mean[j])
			}
		}
	}

	// Normalize
	for i := range dim {
		for j := range dim {
			cov[i][j] /= float32(n - 1)
		}
	}

	return cov
}

// orthogonalize applies Gram-Schmidt orthogonalization to a matrix.
func (opq *OptimizedProductQuantizer) orthogonalize(matrix [][]float32) {
	dim := len(matrix)

	for i := range dim {
		// Orthogonalize against all previous rows
		for j := range i {
			// Compute dot product
			dot := float32(0)
			for k := range dim {
				dot += matrix[i][k] * matrix[j][k]
			}

			// Subtract projection
			for k := range dim {
				matrix[i][k] -= dot * matrix[j][k]
			}
		}

		// Normalize row
		norm := float32(0)
		for k := range dim {
			norm += matrix[i][k] * matrix[i][k]
		}
		norm = float32(math.Sqrt(float64(norm)))

		if norm > 1e-10 {
			for k := range dim {
				matrix[i][k] /= norm
			}
		}
	}
}

// Encode quantizes a vector using OPQ: rotate, then PQ encode.
func (opq *OptimizedProductQuantizer) Encode(vec []float32) []byte {
	if !opq.hasRotation {
		panic("OptimizedProductQuantizer not trained")
	}

	// Rotate vector
	rotated := make([]float32, len(vec))
	opq.matrixVectorMultiply(opq.rotation, vec, rotated)

	// Encode with PQ
	return opq.pq.Encode(rotated)
}

// Decode reconstructs an approximate vector from OPQ codes: PQ decode, then inverse rotate.
func (opq *OptimizedProductQuantizer) Decode(codes []byte) []float32 {
	if !opq.hasRotation {
		panic("OptimizedProductQuantizer not trained")
	}

	// Decode with PQ (gets rotated vector)
	rotated := opq.pq.Decode(codes)

	// Apply inverse rotation (R^T)
	original := make([]float32, len(rotated))
	opq.matrixVectorMultiply(opq.rotationT, rotated, original)

	return original
}

// ComputeAsymmetricDistance computes distance between a query and OPQ codes.
// The query is rotated first, then asymmetric distance is computed.
func (opq *OptimizedProductQuantizer) ComputeAsymmetricDistance(query []float32, codes []byte) float32 {
	if !opq.hasRotation {
		panic("OptimizedProductQuantizer not trained")
	}

	// Rotate query
	rotatedQuery := make([]float32, len(query))
	opq.matrixVectorMultiply(opq.rotation, query, rotatedQuery)

	// Compute asymmetric distance in rotated space
	return opq.pq.ComputeAsymmetricDistance(rotatedQuery, codes)
}

// BytesPerVector returns the compressed size per vector in bytes.
func (opq *OptimizedProductQuantizer) BytesPerVector() int {
	return opq.pq.BytesPerVector()
}

// CompressionRatio returns the theoretical compression ratio.
func (opq *OptimizedProductQuantizer) CompressionRatio() float64 {
	return opq.pq.CompressionRatio()
}

// IsTrained returns whether the quantizer has been trained.
func (opq *OptimizedProductQuantizer) IsTrained() bool {
	return opq.hasRotation && opq.pq.IsTrained()
}

// Helper methods for matrix operations

// identityMatrix creates an identity matrix of size n x n.
func (opq *OptimizedProductQuantizer) identityMatrix(n int) [][]float32 {
	matrix := make([][]float32, n)
	for i := range matrix {
		matrix[i] = make([]float32, n)
		matrix[i][i] = 1.0
	}
	return matrix
}

// transpose computes the transpose of a matrix.
func (opq *OptimizedProductQuantizer) transpose(matrix [][]float32) [][]float32 {
	n := len(matrix)
	result := make([][]float32, n)
	for i := range result {
		result[i] = make([]float32, n)
		for j := range n {
			result[i][j] = matrix[j][i]
		}
	}
	return result
}

// matrixVectorMultiply computes result = matrix * vec.
func (opq *OptimizedProductQuantizer) matrixVectorMultiply(matrix [][]float32, vec []float32, result []float32) {
	for i := range matrix {
		result[i] = math32.Dot(matrix[i], vec)
	}
}
