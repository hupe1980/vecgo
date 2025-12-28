package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/quantization"
)

func main() {
	fmt.Println("=== Vecgo Quantization Example ===\n")

	// Configuration
	const (
		numVectors = 10000
		dimensions = 128
		k          = 10
	)

	// Generate random vectors
	fmt.Printf("Generating %d random vectors with %d dimensions...\n", numVectors, dimensions)
	vectors := generateRandomVectors(numVectors, dimensions)

	// Create quantizer and train it on the dataset
	fmt.Println("\nTraining 8-bit scalar quantizer...")
	quantizer := quantization.NewScalarQuantizer()
	if err := quantizer.Train(vectors); err != nil {
		log.Fatalf("Failed to train quantizer: %v", err)
	}

	fmt.Printf("Quantizer trained: min=%.4f, max=%.4f\n", quantizer.Min(), quantizer.Max())
	fmt.Printf("Compression ratio: %.1fx (float32 → uint8)\n", quantizer.CompressionRatio())
	fmt.Printf("Estimated quantization error: %.6f per dimension\n", quantizer.QuantizationError())

	// Calculate memory usage
	uncompressedSize := numVectors * dimensions * 4 // float32 = 4 bytes
	compressedSize := numVectors * dimensions * 1   // uint8 = 1 byte
	memorySaved := uncompressedSize - compressedSize

	fmt.Printf("\nMemory comparison:\n")
	fmt.Printf("  Uncompressed: %.2f MB\n", float64(uncompressedSize)/(1024*1024))
	fmt.Printf("  Compressed:   %.2f MB\n", float64(compressedSize)/(1024*1024))
	fmt.Printf("  Saved:        %.2f MB (%.1f%% reduction)\n",
		float64(memorySaved)/(1024*1024),
		100.0*float64(memorySaved)/float64(uncompressedSize))

	// Demonstrate quantization encode/decode
	fmt.Println("\n=== Quantization Quality Test ===")
	testVector := vectors[0]
	encoded := quantizer.Encode(testVector)
	decoded := quantizer.Decode(encoded)

	// Calculate reconstruction error
	var sumSquaredError float32
	var maxError float32
	for i := range testVector {
		err := testVector[i] - decoded[i]
		squaredErr := err * err
		sumSquaredError += squaredErr
		absErr := abs(err)
		if absErr > maxError {
			maxError = absErr
		}
	}
	mse := sumSquaredError / float32(len(testVector))

	fmt.Printf("Sample vector reconstruction:\n")
	fmt.Printf("  Original:  [%.4f, %.4f, %.4f, ..., %.4f]\n",
		testVector[0], testVector[1], testVector[2], testVector[len(testVector)-1])
	fmt.Printf("  Decoded:   [%.4f, %.4f, %.4f, ..., %.4f]\n",
		decoded[0], decoded[1], decoded[2], decoded[len(decoded)-1])
	fmt.Printf("  MSE:       %.6f\n", mse)
	fmt.Printf("  Max Error: %.6f\n", maxError)

	// Create HNSW index
	fmt.Println("\n=== Building HNSW Index ===")
	db, err := vecgo.HNSW[int](dimensions).
		SquaredL2().
		M(16).
		EF(200).
		Build()
	if err != nil {
		log.Fatalf("Failed to create vecgo: %v", err)
	}

	ctx := context.Background()

	// Insert vectors
	fmt.Printf("Inserting %d vectors...\n", numVectors)
	start := time.Now()
	for i, vec := range vectors {
		_, err := db.Insert(ctx, vecgo.VectorWithData[int]{
			Vector: vec,
			Data:   i,
		})
		if err != nil {
			log.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}
	insertDuration := time.Since(start)
	fmt.Printf("Inserted %d vectors in %v (%.0f vectors/sec)\n",
		numVectors, insertDuration, float64(numVectors)/insertDuration.Seconds())

	// Perform search
	fmt.Println("\n=== Search Performance ===")
	queryVector := generateRandomVector(dimensions)

	start = time.Now()
	results, err := db.KNNSearch(ctx, queryVector, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	searchDuration := time.Since(start)

	fmt.Printf("Found %d nearest neighbors in %v\n", len(results), searchDuration)
	fmt.Println("\nTop 5 results:")
	maxResults := 5
	if len(results) < maxResults {
		maxResults = len(results)
	}
	for i := 0; i < maxResults; i++ {
		fmt.Printf("  %d. ID=%d, Distance=%.4f\n", i+1, results[i].ID, results[i].Distance)
	}

	// Summary
	fmt.Println("\n=== Quantization Benefits ===")
	fmt.Println("✓ 4x memory reduction (float32 → uint8)")
	fmt.Println("✓ Faster distance computations (integer arithmetic)")
	fmt.Println("✓ Better cache utilization (smaller vectors)")
	fmt.Println("✓ Minimal accuracy loss (<1% typical)")
	fmt.Println("\nNote: Quantization primitives are ready. Full integration with index storage")
	fmt.Println("can be added based on specific requirements.")
}

func generateRandomVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = generateRandomVector(dim)
	}
	return vectors
}

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // Range: [-1, 1]
	}
	return vec
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
