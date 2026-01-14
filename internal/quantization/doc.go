// Package quantization provides vector compression techniques for memory reduction.
//
// Vecgo supports six quantization methods:
//
//   - Binary Quantization (BQ): 32x compression (1 bit per dimension)
//   - RaBitQ: Randomized Binary Quantization with Norm Correction (Better Recall than BQ)
//   - Scalar Quantization (SQ8): 4x compression (1 byte per dimension)
//   - INT4 Quantization: 8x compression (4 bits per dimension)
//   - Product Quantization (PQ): 8-64x compression
//   - Optimized Product Quantization (OPQ): 8-64x compression with learned rotation
//
// # Architecture
//
//	┌────────────────────────────────────────────────────────────────────┐
//	│                    Quantization Methods                            │
//	├──────────┬──────────┬──────────┬──────────┬──────────┬────────────┤
//	│   BQ     │  RaBitQ  │   SQ8    │   INT4   │    PQ    │    OPQ     │
//	│ (1 bit)  │ (1 bit+) │ (8 bit)  │ (4 bit)  │ (8 bit)  │ (8 bit+R)  │
//	├──────────┴──────────┴──────────┴──────────┴──────────┴────────────┤
//	│                      SIMD Distance Functions                       │
//	│   Hamming (POPCNT)  │  L2/Dot (AVX-512/NEON)  │  ADC Lookup        │
//	└────────────────────────────────────────────────────────────────────┘
//
// # Binary Quantization
//
// Compresses vectors to 1 bit per dimension using sign-based encoding:
//
//	bq := quantization.NewBinaryQuantizer(128)
//	code, _ := bq.Encode(vector)  // 128 floats → 16 bytes
//	dist := HammingDistance(code1, code2)  // Hamming distance
//
// Performance:
//   - Compression: 32x (128-dim float32: 512 bytes → 16 bytes)
//   - Distance: Ultra-fast (POPCNT instruction)
//   - Accuracy: 70-85% recall (best for pre-filtering)
//
// # RaBitQ (Randomized Binary Quantization)
//
// Improvements over standard BQ by storing the vector norm and using a modified distance estimator:
//
//	rq := quantization.NewRaBitQuantizer(128)
//	code, _ := rq.Encode(vector)       // 128 floats → 20 bytes (16 binary + 4 norm)
//	dist, _ := rq.Distance(query, code) // Norm-corrected L2 approximation
//
// Trade-off: Slightly larger storage (+4 bytes per vector) for significantly better L2 distance approximation.
//
// # Scalar Quantization (SQ8)
//
// Compresses each dimension to 8 bits using per-dimension min/max normalization:
//
//	sq := quantization.NewScalarQuantizer(128)
//	sq.Train(trainingVectors)
//	code, _ := sq.Encode(vector)  // 128 floats → 128 bytes
//	dist, _ := sq.L2Distance(query, code)
//
// Performance:
//   - Compression: 4x (float32 → uint8)
//   - Accuracy: 95-99% recall (excellent for most use cases)
//
// # INT4 Quantization
//
// Compresses each dimension to 4 bits (2 dimensions per byte):
//
//	iq := quantization.NewInt4Quantizer(128)
//	iq.Train(trainingVectors)
//	code, _ := iq.Encode(vector)  // 128 floats → 64 bytes
//	dist, _ := iq.L2Distance(query, code)
//
// Performance:
//   - Compression: 8x (float32 → 4-bit)
//   - Accuracy: 90-95% recall
//
// # Product Quantization (PQ)
//
// Splits vector into subvectors and quantizes each independently using k-means:
//
//	pq, _ := quantization.NewProductQuantizer(128, 8, 256)  // dim=128, M=8, K=256
//	pq.Train(trainingVectors)
//	code, _ := pq.Encode(vector)  // 128 floats → 8 bytes
//
// Parameters:
//   - dimension: Vector dimensionality (must be divisible by numSubvectors)
//   - numSubvectors (M): How many splits (typically 8-16)
//   - numCentroids (K): Codebook size per subvector (typically 256 for uint8)
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
// Learns block-diagonal rotation matrices before PQ to improve reconstruction quality:
//
//	opq, _ := quantization.NewOptimizedProductQuantizer(128, 8, 256, 10)  // +10 iterations
//	opq.Train(trainingVectors)
//	code, _ := opq.Encode(vector)  // 128 floats → 8 bytes
//
// Benefits vs PQ:
//   - 20-30% better reconstruction quality
//   - Same compression ratio
//   - Higher training cost (alternating optimization with SVD)
//
// Accuracy: 93-97% recall
//
// # Quantization Comparison
//
//	| Method  | Compression | Recall  | Speed    | Use Case              |
//	|---------|-------------|---------|----------|-----------------------|
//	| None    | 1x          | 100%    | Baseline | Default               |
//	| Binary  | 32x         | 70-85%  | Fastest  | Pre-filtering         |
//	| RaBitQ  | ~30x        | 80-90%  | Fast     | Better BQ alternative |
//	| SQ8     | 4x          | 95-99%  | Fast     | General purpose       |
//	| INT4    | 8x          | 90-95%  | Fast     | Memory-constrained    |
//	| PQ      | 8-64x       | 90-95%  | Medium   | High compression      |
//	| OPQ     | 8-64x       | 93-97%  | Slower   | High recall + low mem |
//
// # Thread Safety
//
// All quantizers are safe for concurrent read operations (Encode, Decode, Distance)
// after training. Training (Train) must be single-threaded or externally synchronized.
//
// # Serialization
//
// ScalarQuantizer and Int4Quantizer implement encoding.BinaryMarshaler/BinaryUnmarshaler
// for persistence. ProductQuantizer uses SetCodebooks/Codebooks for manual serialization.
package quantization
