// Package index provides vector index interfaces and implementations.
//
// Vecgo supports three index types:
//
//   - Flat: Exact nearest neighbor search (brute-force with SIMD optimization)
//   - HNSW: Hierarchical Navigable Small World graph for fast approximate search
//   - DiskANN: Disk-resident Vamana graph for billion-scale datasets
//
// # Index Selection
//
// Choose based on dataset size and accuracy requirements:
//
//   - Flat: <100K vectors, 100% recall required
//   - HNSW: 100K-10M vectors, 95-99% recall acceptable
//   - DiskANN: 10M+ vectors, 90-95% recall, limited RAM
//
// # Distance Types
//
// All indexes support three distance functions:
//
//   - DistanceTypeSquaredL2: Squared Euclidean distance (most common)
//   - DistanceTypeCosine: Cosine similarity (vectors are normalized)
//   - DistanceTypeDotProduct: Dot product (represented as negative for "lower is better")
//
// # Index Interface
//
// All index implementations satisfy the core Index interface:
//
//	type Index interface {
//	    Dimension() int
//	    DistanceType() DistanceType
//	    Add(vector []float32) (uint64, error)
//	    Search(query []float32, k int) ([]Result, error)
//	    Update(id uint64, vector []float32) error
//	    Delete(id uint64) error
//	    Close() error
//	}
//
// # Quantization
//
// Indexes support quantization for memory reduction:
//
//   - Binary Quantization: 32x compression (1 bit per dimension)
//   - Product Quantization (PQ): 8-64x compression
//   - Optimized PQ (OPQ): 8-64x compression with better accuracy
//
// See package quantization for details.
//
// # Subpackages
//
//   - flat: Exact search with columnar storage
//   - hnsw: Approximate search with graph index
//   - diskann: Disk-resident search for billion-scale datasets
package index
