// Package quantization provides vector compression techniques for memory reduction.
//
// Vecgo supports three quantization methods:
//
//   - Binary Quantization: 32x compression (1 bit per dimension)
//   - Product Quantization (PQ): 8-64x compression
//   - Optimized Product Quantization (OPQ): 8-64x compression with better recall
//
// # Binary Quantization
//
// Compresses vectors to 1 bit per dimension using sign-based encoding:
//
//	quantizer := quantization.NewBinaryQuantizer(128)
//	code := quantizer.Encode(vector)  // 128 floats → 16 bytes
//	distance := quantizer.Distance(code1, code2)  // Hamming distance
//
// Performance:
//   - Compression: 32x (128-dim float32: 512 bytes → 16 bytes)
//   - Distance: 0.68ns for 128-dim (ultra-fast)
//   - Accuracy: 70-85% recall (suitable for pre-filtering)
//
// # Product Quantization (PQ)
//
// Splits vector into subvectors and quantizes each independently:
//
//	pq, err := quantization.NewPQ(
//	    vectors,           // Training data
//	    128,               // Dimension
//	    8,                 // Number of subvectors
//	    256,               // Centroids per subvector
//	    distance.SquaredL2, // Distance type
//	)
//	code := pq.Encode(vector)  // 128 floats → 8 bytes
//
// Parameters:
//   - subvectors: How many splits (typically 8-16)
//   - centroids: Codebook size per subvector (typically 256)
//
// Memory reduction:
//   - 128-dim float32 = 512 bytes
//   - PQ(8, 256) = 8 bytes (64x compression)
//   - PQ(16, 256) = 16 bytes (32x compression)
//
// Accuracy: 90-95% recall
//
// # Optimized Product Quantization (OPQ)
//
// Learns a rotation matrix before PQ to improve reconstruction quality:
//
//	opq, err := quantization.NewOPQ(
//	    vectors,           // Training data
//	    128,               // Dimension
//	    8,                 // Number of subvectors
//	    256,               // Centroids per subvector
//	    distance.SquaredL2, // Distance type
//	)
//	code := opq.Encode(vector)  // 128 floats → 8 bytes
//
// Benefits vs PQ:
//   - 20-30% better reconstruction quality
//   - Same compression ratio
//   - Slightly higher encoding cost (rotation + quantization)
//
// Accuracy: 93-97% recall
//
// # Quantization Comparison
//
//	| Method  | Compression | Recall  | Speed    | Use Case              |
//	|---------|-------------|---------|----------|-----------------------|
//	| None    | 1x          | 100%    | Baseline | Default               |
//	| Binary  | 32x         | 70-85%  | Fastest  | Pre-filtering         |
//	| PQ      | 8-64x       | 90-95%  | Fast     | Memory-constrained    |
//	| OPQ     | 8-64x       | 93-97%  | Medium   | High recall + low mem |
//
// # Usage in Vecgo (current)
//
// The current engine-first public facade (`vecgo.Open(...)`) does not expose end-user
// quantization knobs yet. Quantization is used internally by on-disk segment writers/readers
// (notably `segment/flat`) when producing or reading quantized Flat files.
//
// Low-level usage (library-style) looks like:
//
//	// SQ8 (scalar) quantization
//	sq := quantization.NewScalarQuantizer(128)
//	_ = sq.Train(trainingVectors)
//	code := sq.Encode(vec)
//	_ = code
//
//	// PQ / OPQ
//	pq, _ := quantization.NewProductQuantizer(128, 8, 256)
//	_ = pq.Train(trainingVectors)
//
//	opq, _ := quantization.NewOptimizedProductQuantizer(128, 8, 256, 10)
//	_ = opq.Train(trainingVectors)
//
// See docs/tuning.md and REFACTORING.md for the current integration status.
package quantization
