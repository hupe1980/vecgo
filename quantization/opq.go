// Package quantization provides advanced vector quantization for extreme compression.
package quantization

import (
	"errors"
	"fmt"
	"sync"

	"github.com/hupe1980/vecgo/internal/simd"
)

// OptimizedProductQuantizer implements Block Optimized Product Quantization (OPQ).
// It splits vectors into subspaces (blocks) and learns an optimal rotation for each block
// to minimize quantization error. This is a "Parametric" OPQ implementation where
// the rotation matrix is constrained to be block-diagonal.
type OptimizedProductQuantizer struct {
	pq            *ProductQuantizer
	rotations     [][][]float32 // [numBlocks][blockSize][blockSize]
	blockSize     int
	numIterations int
	trained       bool

	// Pools for temporary buffers to avoid allocations on hot paths
	vecPool *sync.Pool
}

// NewOptimizedProductQuantizer creates a new OPQ quantizer.
// It automatically selects a block size for the rotation matrices.
func NewOptimizedProductQuantizer(dimension, numSubvectors, numCentroids, numIterations int) (*OptimizedProductQuantizer, error) {
	pq, err := NewProductQuantizer(dimension, numSubvectors, numCentroids)
	if err != nil {
		return nil, err
	}

	subvectorSize := dimension / numSubvectors
	if dimension%numSubvectors != 0 {
		return nil, errors.New("dimension must be divisible by numSubvectors")
	}

	// Determine block size.
	// We want blocks to be multiples of subvectorSize and roughly 32-64 dimensions.
	// If dimension is small, use full dimension.
	blockSize := dimension
	if dimension > 64 {
		// Try to find a multiple of subvectorSize close to 32
		target := 32
		bestDiff := 1000

		// Search for divisors of dimension that are multiples of subvectorSize
		for b := subvectorSize; b <= dimension; b += subvectorSize {
			if dimension%b == 0 {
				diff := abs(b - target)
				if diff < bestDiff {
					bestDiff = diff
					blockSize = b
				}
			}
		}
	}

	numBlocks := dimension / blockSize
	rotations := make([][][]float32, numBlocks)
	for i := range rotations {
		rotations[i] = identityMatrix(blockSize)
	}

	return &OptimizedProductQuantizer{
		pq:            pq,
		rotations:     rotations,
		blockSize:     blockSize,
		numIterations: numIterations,
		trained:       false,
		vecPool: &sync.Pool{
			New: func() interface{} {
				s := make([]float32, dimension)
				return &s
			},
		},
	}, nil
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Train calibrates the OPQ quantizer using alternating optimization.
func (opq *OptimizedProductQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided for training")
	}
	if len(vectors[0]) != opq.pq.dimension {
		return errors.New("vector dimension mismatch")
	}

	// Initialize rotations to identity
	for i := range opq.rotations {
		opq.rotations[i] = identityMatrix(opq.blockSize)
	}
	opq.trained = true

	dim := opq.pq.dimension
	rotatedVectors := make([][]float32, len(vectors))
	for i := range rotatedVectors {
		rotatedVectors[i] = make([]float32, dim)
	}

	// Alternating optimization
	for iter := 0; iter < opq.numIterations; iter++ {
		// Step 1: Rotate vectors
		// Parallelize rotation
		// For simplicity in this implementation, we do it serially or simple parallel loop
		// Since we are in Train, performance is less critical than Encode, but still important.
		for i, vec := range vectors {
			opq.rotateVector(vec, rotatedVectors[i])
		}

		// Step 2: Train PQ
		if err := opq.pq.Train(rotatedVectors); err != nil {
			return err
		}

		// Step 3: Update rotations (Procrustes)
		// We need to find R that minimizes || XR - \hat{Y} ||^2
		// where X is original vectors, \hat{Y} is reconstructed vectors (from PQ centroids).
		// Note: PQ centroids are in the rotated space.
		// So \hat{Y} are the centroids corresponding to rotated vectors.
		// We want XR \approx \hat{Y}.
		// M = X^T * \hat{Y}

		// Compute \hat{Y} (reconstructed vectors)
		// We can do this block-wise to save memory.
		// For each block b:
		//   M_b = sum_i (X_{i,b}^T * \hat{Y}_{i,b})
		//   R_b = Procrustes(M_b)

		// Accumulate M matrices
		mMatrices := make([][][]float32, len(opq.rotations))
		for b := range mMatrices {
			mMatrices[b] = make([][]float32, opq.blockSize)
			for r := range mMatrices[b] {
				mMatrices[b][r] = make([]float32, opq.blockSize)
			}
		}

		// Iterate over vectors
		for i, vec := range vectors {
			// Encode/Decode to get \hat{Y} (in rotated space)
			// We can optimize this by just finding the nearest centroid for each subvector
			// without full encode/decode byte array.
			// But using Encode/Decode is safer/easier.
			encoded, err := opq.pq.Encode(rotatedVectors[i])
			if err != nil {
				return err
			}
			reconstructed, err := opq.pq.Decode(encoded) // \hat{Y}
			if err != nil {
				return err
			}

			// Accumulate X^T * \hat{Y} for each block
			for b := 0; b < len(opq.rotations); b++ {
				start := b * opq.blockSize
				end := start + opq.blockSize

				xBlock := vec[start:end]
				yBlock := reconstructed[start:end]

				// M_b += xBlock^T * yBlock
				// M_b[r][c] += xBlock[r] * yBlock[c]
				for r := 0; r < opq.blockSize; r++ {
					xr := xBlock[r]
					row := mMatrices[b][r]
					for c := 0; c < opq.blockSize; c++ {
						row[c] += xr * yBlock[c]
					}
				}
			}
		}

		// Solve Procrustes for each block
		for b := range opq.rotations {
			rot, err := computeProcrustesRotation(mMatrices[b])
			if err != nil {
				return fmt.Errorf("procrustes rotation for block %d: %w", b, err)
			}
			opq.rotations[b] = rot
		}
	}

	return nil
}

// rotateVector applies the block-diagonal rotation to src and stores in dst.
func (opq *OptimizedProductQuantizer) rotateVector(src, dst []float32) {
	for b, rotation := range opq.rotations {
		start := b * opq.blockSize
		// Block matrix multiplication
		// dst[start:start+bs] = rotation * src[start:start+bs]

		for i := 0; i < opq.blockSize; i++ {
			row := rotation[i]
			// Unroll loop slightly for performance?
			// Or rely on compiler.
			// src slice
			srcBlock := src[start : start+opq.blockSize]

			// Dot product
			sum := simd.Dot(row, srcBlock)
			dst[start+i] = sum
		}
	}
}

// Encode quantizes a vector using OPQ.
func (opq *OptimizedProductQuantizer) Encode(vec []float32) ([]byte, error) {
	if !opq.trained {
		return nil, errors.New("OptimizedProductQuantizer not trained")
	}

	// Get temporary buffer
	rotatedPtr := opq.vecPool.Get().(*[]float32)
	rotated := *rotatedPtr
	defer opq.vecPool.Put(rotatedPtr)

	opq.rotateVector(vec, rotated)
	return opq.pq.Encode(rotated)
}

// Decode reconstructs a vector.
func (opq *OptimizedProductQuantizer) Decode(codes []byte) ([]float32, error) {
	if !opq.trained {
		return nil, errors.New("OptimizedProductQuantizer not trained")
	}

	// Decode in rotated space
	rotated, err := opq.pq.Decode(codes)
	if err != nil {
		return nil, err
	}

	// Inverse rotate (R^T)
	// Since R is block diagonal orthogonal, R^T is just block transpose.
	original := make([]float32, len(rotated))

	for b, rotation := range opq.rotations {
		start := b * opq.blockSize

		// dst = R^T * src
		// dst[i] = sum(R[j][i] * src[j])

		srcBlock := rotated[start : start+opq.blockSize]
		dstBlock := original[start : start+opq.blockSize]

		for i := 0; i < opq.blockSize; i++ {
			sum := float32(0)
			for j := 0; j < opq.blockSize; j++ {
				sum += rotation[j][i] * srcBlock[j]
			}
			dstBlock[i] = sum
		}
	}

	return original, nil
}

// ComputeAsymmetricDistance computes distance between a query and OPQ codes.
func (opq *OptimizedProductQuantizer) ComputeAsymmetricDistance(query []float32, codes []byte) (float32, error) {
	if !opq.trained {
		return 0, errors.New("OptimizedProductQuantizer not trained")
	}

	// Rotate query once
	rotatedQueryPtr := opq.vecPool.Get().(*[]float32)
	rotatedQuery := *rotatedQueryPtr
	defer opq.vecPool.Put(rotatedQueryPtr)

	opq.rotateVector(query, rotatedQuery)

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

// IsTrained returns true if trained.
func (opq *OptimizedProductQuantizer) IsTrained() bool {
	return opq.trained
}

// identityMatrix creates an identity matrix.
func identityMatrix(n int) [][]float32 {
	matrix := make([][]float32, n)
	for i := range matrix {
		matrix[i] = make([]float32, n)
		matrix[i][i] = 1.0
	}
	return matrix
}
