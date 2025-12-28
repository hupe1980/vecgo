package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/quantization"
)

func main() {
	ctx := context.Background()

	// Configuration
	dimension := 128
	numVectors := 10000
	numSubvectors := 8
	numCentroids := 256
	opqIterations := 10

	fmt.Println("=== Optimized Product Quantization (OPQ) Demo ===\n")

	// Create HNSW index
	db, err := vecgo.HNSW[string](dimension).SquaredL2().Build()
	if err != nil {
		log.Fatal(err)
	}

	// Generate training vectors for OPQ
	fmt.Printf("Generating %d training vectors (dim=%d)...\n", numVectors, dimension)
	trainingVectors := make([][]float32, numVectors)
	for i := range trainingVectors {
		trainingVectors[i] = generateRandomVector(dimension, i)
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

	// Enable PQ on the index (note: OPQ is not yet integrated into the index layer)
	// For now, we just demonstrate that PQ can be enabled
	fmt.Println("\n=== Enabling PQ on HNSW Index ===")
	err = db.EnableProductQuantization(index.ProductQuantizationConfig{
		NumSubvectors: numSubvectors,
		NumCentroids:  numCentroids,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("✓ Product Quantization enabled on index")

	// Insert vectors
	fmt.Printf("\nInserting %d vectors...\n", 1000)
	for i := 0; i < 1000; i++ {
		vec := generateRandomVector(dimension, i+numVectors)
		_, err := db.Insert(ctx, vecgo.VectorWithData[string]{
			Vector: vec,
			Data:   fmt.Sprintf("vec_%d", i),
		})
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println("✓ Insert complete")

	// Perform searches
	fmt.Println("\n=== Search with PQ Acceleration ===")
	query := generateRandomVector(dimension, 99999)

	results, err := db.KNNSearch(ctx, query, 10, func(o *vecgo.KNNSearchOptions) {
		o.EF = 50
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Found %d results:\n", len(results))
	for i, result := range results {
		fmt.Printf("  %d. %s (distance: %.4f)\n", i+1, result.Data, result.Distance)
	}

	// Show index statistics
	fmt.Println("\n=== Index Statistics ===")
	stats := db.Stats()

	// Extract relevant stats
	if totalVec, ok := stats.Storage["vectors"]; ok {
		fmt.Printf("Total vectors:     %s\n", totalVec)
	}
	if memUsage, ok := stats.Storage["memory_usage"]; ok {
		fmt.Printf("Memory usage:      %s\n", memUsage)
	}
	fmt.Printf("PQ enabled:        true\n")

	fmt.Println("\n✓ Demo complete!")
	fmt.Println("\nNote: This example demonstrates OPQ training and quality improvement.")
	fmt.Println("Full OPQ integration into the index layer is a future enhancement.")
}

// generateRandomVector creates a random vector with deterministic seeding
func generateRandomVector(dim, seed int) []float32 {
	rng := rand.New(rand.NewSource(int64(seed)))
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1 // [-1, 1]
	}

	// Normalize to unit length for more realistic embeddings
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

	return vec
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
