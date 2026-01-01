package main

import (
	"fmt"
	"log"
	"math"

	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/testutil"
)

func main() {
	// Configuration
	dimension := 128
	numVectors := 10000
	numSubvectors := 8
	numCentroids := 256
	opqIterations := 10

	fmt.Println("=== Optimized Product Quantization (OPQ) Demo ===")
	fmt.Println()

	// Generate training vectors for OPQ
	fmt.Printf("Generating %d training vectors (dim=%d)...\n", numVectors, dimension)
	rng := testutil.NewRNG(0)
	trainingVectors := rng.UniformVectors(numVectors, dimension)

	// Normalize vectors (OPQ works best with normalized data for cosine/L2)
	for _, vec := range trainingVectors {
		normalize(vec)
	}

	// Train OPQ quantizer
	fmt.Printf("\nTraining OPQ quantizer (M=%d, K=%d, iterations=%d)...\n",
		numSubvectors, numCentroids, opqIterations)
	opq, err := quantization.NewOptimizedProductQuantizer(
		dimension,
		numSubvectors,
		numCentroids,
		opqIterations,
	)
	if err != nil {
		log.Fatal(err)
	}

	if err := opq.Train(trainingVectors); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✓ OPQ training complete\n")

	// Train standard PQ for comparison
	fmt.Printf("\nTraining standard PQ quantizer for comparison...\n")
	pq, err := quantization.NewProductQuantizer(dimension, numSubvectors, numCentroids)
	if err != nil {
		log.Fatal(err)
	}
	if err := pq.Train(trainingVectors); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("✓ PQ training complete\n")

	// Measure reconstruction quality
	fmt.Println("\n=== Reconstruction Quality Comparison ===")
	testVectors := trainingVectors[:100]

	opqError := float32(0)
	pqError := float32(0)

	for _, vec := range testVectors {
		// OPQ reconstruction
		opqCodes := opq.Encode(vec)
		opqRecon := opq.Decode(opqCodes)
		opqError += l2DistanceSquared(vec, opqRecon)

		// PQ reconstruction
		pqCodes := pq.Encode(vec)
		pqRecon := pq.Decode(pqCodes)
		pqError += l2DistanceSquared(vec, pqRecon)
	}

	opqError /= float32(len(testVectors))
	pqError /= float32(len(testVectors))

	improvement := (pqError - opqError) / pqError * 100

	fmt.Printf("Standard PQ avg error:  %.6f\n", pqError)
	fmt.Printf("OPQ avg error:          %.6f\n", opqError)
	fmt.Printf("Improvement:            %.1f%%\n", improvement)

	// Show compression statistics
	fmt.Println("\n=== Compression Statistics ===")
	originalBytes := dimension * 4 // float32 = 4 bytes
	compressedBytes := opq.BytesPerVector()
	ratio := opq.CompressionRatio()

	fmt.Printf("Original size:     %d bytes\n", originalBytes)
	fmt.Printf("Compressed size:   %d bytes\n", compressedBytes)
	fmt.Printf("Compression ratio: %.1fx\n", ratio)

	fmt.Println("\n✓ Demo complete!")
	fmt.Println("\nNote: This example demonstrates OPQ training and quality improvement.")
	fmt.Println("Full OPQ integration into the index layer is a future enhancement.")
}

// normalize normalizes a vector to unit length
func normalize(vec []float32) {
	norm := float32(0)
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}

// l2DistanceSquared computes squared L2 distance
func l2DistanceSquared(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func generateRandomVector(dim int, seed int) []float32 {
	rng := testutil.NewRNG(int64(seed))
	vec := rng.UniformVectors(1, dim)[0]
	normalize(vec)
	return vec
}
