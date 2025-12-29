package diskann

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/quantization"
)

// DefaultOptions returns sensible defaults for DiskANN.
func DefaultOptions() *Options {
	return &Options{
		R:                    64,   // Max edges per node
		L:                    100,  // Build list size
		Alpha:                1.2,  // Pruning factor
		PQSubvectors:         32,   // Number of PQ subvectors
		PQCentroids:          256,  // Centroids per subspace
		NumBuildShards:       1,    // Sequential build by default
		BeamWidth:            4,    // Parallel disk reads
		RerankK:              100,  // Candidates to rerank
		EnableAutoCompaction: true, // Auto compaction enabled
		CompactionThreshold:  0.2,  // Compact when 20% deleted
		CompactionInterval:   300,  // Check every 5 minutes
		CompactionMinVectors: 1000, // Don't compact if fewer than 1000 vectors
	}
}

// Options configures DiskANN index construction and search.
type Options struct {
	// R is the maximum number of edges per node in the Vamana graph.
	// Higher R increases recall but uses more memory. Typical: 32-128.
	R int

	// L is the size of the candidate list during graph construction.
	// Higher L improves graph quality but slows construction. Typical: 100-200.
	L int

	// Alpha is the pruning factor for edge selection (>= 1.0).
	// Higher Alpha keeps more diverse edges. Typical: 1.0-1.5.
	Alpha float32

	// PQSubvectors is the number of subvectors for PQ compression (M).
	// Dimension must be divisible by this. Typical: 8-64.
	PQSubvectors int

	// PQCentroids is the number of centroids per subspace (K).
	// Typically 256 for uint8 codes.
	PQCentroids int

	// NumBuildShards parallelizes graph construction.
	// Each shard builds independently then merges. Typical: 1-8.
	NumBuildShards int

	// BeamWidth controls parallel disk reads during search.
	// Higher values improve throughput at cost of latency. Typical: 1-8.
	BeamWidth int

	// RerankK is the number of candidates to fetch from disk for reranking.
	// Must be >= k in search. Typical: 50-200.
	RerankK int

	// EnableAutoCompaction enables background compaction to remove deleted vectors.
	// Compaction rebuilds the graph and re-trains PQ for optimal performance.
	EnableAutoCompaction bool

	// CompactionThreshold is the fraction of deleted vectors that triggers compaction.
	// For example, 0.2 means compact when 20% of vectors are deleted. Typical: 0.1-0.3.
	CompactionThreshold float32

	// CompactionInterval is the interval in seconds between compaction checks.
	// Typical: 60-600 (1-10 minutes).
	CompactionInterval int

	// CompactionMinVectors is the minimum number of vectors before compaction is considered.
	// Prevents compacting tiny indexes. Typical: 100-10000.
	CompactionMinVectors int
}

// Builder constructs a DiskANN index from vectors.
type Builder struct {
	opts      *Options
	dim       int
	distType  index.DistanceType
	distFunc  index.DistanceFunc
	indexPath string

	// In-memory data during construction
	vectors    [][]float32 // All vectors (temporary)
	graph      [][]uint32  // Adjacency lists
	pq         *quantization.ProductQuantizer
	pqCodes    [][]byte // PQ codes for all vectors
	entryPoint uint32   // Entry point for search

	mu sync.Mutex
}

// NewBuilder creates a new DiskANN index builder.
func NewBuilder(dim int, distType index.DistanceType, indexPath string, opts *Options) (*Builder, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	if dim <= 0 {
		return nil, &index.ErrInvalidDimension{Dimension: dim}
	}

	if dim%opts.PQSubvectors != 0 {
		return nil, fmt.Errorf("diskann: dimension %d not divisible by PQSubvectors %d", dim, opts.PQSubvectors)
	}

	if opts.R <= 0 || opts.L <= 0 || opts.Alpha < 1.0 {
		return nil, errors.New("diskann: invalid graph parameters")
	}

	// Ensure directory exists
	if err := os.MkdirAll(indexPath, 0755); err != nil {
		return nil, fmt.Errorf("diskann: create index directory: %w", err)
	}

	return &Builder{
		opts:      opts,
		dim:       dim,
		distType:  distType,
		distFunc:  index.NewDistanceFunc(distType),
		indexPath: indexPath,
		vectors:   make([][]float32, 0, 10000),
		graph:     make([][]uint32, 0, 10000),
	}, nil
}

// Add adds a single vector to the index.
func (b *Builder) Add(vec []float32) (uint32, error) {
	if len(vec) != b.dim {
		return 0, &index.ErrDimensionMismatch{Expected: b.dim, Actual: len(vec)}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	id := uint32(len(b.vectors))

	// Copy vector
	v := make([]float32, len(vec))
	copy(v, vec)
	b.vectors = append(b.vectors, v)

	// Initialize empty adjacency list
	b.graph = append(b.graph, make([]uint32, 0, b.opts.R))

	return id, nil
}

// AddBatch adds multiple vectors to the index.
func (b *Builder) AddBatch(vectors [][]float32) ([]uint32, error) {
	ids := make([]uint32, len(vectors))
	for i, vec := range vectors {
		id, err := b.Add(vec)
		if err != nil {
			return ids[:i], err
		}
		ids[i] = id
	}
	return ids, nil
}

// Build constructs the Vamana graph and writes index files.
func (b *Builder) Build(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := len(b.vectors)
	if n == 0 {
		return errors.New("diskann: no vectors to build")
	}

	// Step 1: Train PQ quantizer
	if err := b.trainPQ(); err != nil {
		return fmt.Errorf("diskann: train PQ: %w", err)
	}

	// Step 2: Encode all vectors to PQ codes
	b.pqCodes = make([][]byte, n)
	for i, vec := range b.vectors {
		b.pqCodes[i] = b.pq.Encode(vec)
	}

	// Step 3: Build Vamana graph
	if err := b.buildVamanaGraph(ctx); err != nil {
		return fmt.Errorf("diskann: build graph: %w", err)
	}

	// Step 4: Write index files
	if err := b.writeIndexFiles(); err != nil {
		return fmt.Errorf("diskann: write files: %w", err)
	}

	return nil
}

// trainPQ trains the product quantizer on all vectors.
func (b *Builder) trainPQ() error {
	var err error
	b.pq, err = quantization.NewProductQuantizer(b.dim, b.opts.PQSubvectors, b.opts.PQCentroids)
	if err != nil {
		return err
	}

	// Use all vectors for training (or sample if too many)
	training := b.vectors
	if len(training) > 100000 {
		// Sample 100K vectors for training
		rng := rand.New(rand.NewSource(42))
		indices := rng.Perm(len(b.vectors))[:100000]
		training = make([][]float32, 100000)
		for i, idx := range indices {
			training[i] = b.vectors[idx]
		}
	}

	return b.pq.Train(training)
}

// buildVamanaGraph constructs the Vamana graph using greedy search and pruning.
func (b *Builder) buildVamanaGraph(ctx context.Context) error {
	n := len(b.vectors)
	R := b.opts.R
	L := b.opts.L
	alpha := b.opts.Alpha

	// Initialize graph with random edges
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		// Add R/2 random edges initially
		edges := make(map[uint32]struct{})
		for len(edges) < R/2 && len(edges) < n-1 {
			j := uint32(rng.Intn(n))
			if j != uint32(i) {
				edges[j] = struct{}{}
			}
		}
		b.graph[i] = make([]uint32, 0, len(edges))
		for j := range edges {
			b.graph[i] = append(b.graph[i], j)
		}
	}

	// Select entry point (centroid approximation)
	b.entryPoint = b.selectEntryPoint()

	// Build graph by iterating over all vectors
	for i := 0; i < n; i++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		// Greedy search from entry point to find neighbors
		neighbors := b.greedySearch(uint32(i), L)

		// Robust prune to select R neighbors
		pruned := b.robustPrune(uint32(i), neighbors, R, alpha)
		b.graph[i] = pruned

		// Add reverse edges
		for _, neighbor := range pruned {
			b.addEdge(neighbor, uint32(i), R, alpha)
		}
	}

	return nil
}

// selectEntryPoint finds a central node as the entry point.
func (b *Builder) selectEntryPoint() uint32 {
	n := len(b.vectors)
	if n == 0 {
		return 0
	}

	// Compute centroid
	centroid := make([]float32, b.dim)
	for _, vec := range b.vectors {
		for j, v := range vec {
			centroid[j] += v
		}
	}
	for j := range centroid {
		centroid[j] /= float32(n)
	}

	// Find nearest vector to centroid
	minDist := float32(math.MaxFloat32)
	entry := uint32(0)
	for i, vec := range b.vectors {
		dist := b.distFunc(centroid, vec)
		if dist < minDist {
			minDist = dist
			entry = uint32(i)
		}
	}

	return entry
}

// greedySearch performs greedy search from entry point to target.
func (b *Builder) greedySearch(target uint32, L int) []uint32 {
	targetVec := b.vectors[target]

	// Min-heap for candidates
	candidates := &distHeap{}
	heap.Init(candidates)

	// Visited set
	visited := make(map[uint32]bool)

	// Start from entry point
	entryDist := b.distFunc(b.vectors[b.entryPoint], targetVec)
	heap.Push(candidates, distNode{id: b.entryPoint, dist: entryDist})
	visited[b.entryPoint] = true

	// Result list (top L closest)
	result := make([]distNode, 0, L)
	result = append(result, distNode{id: b.entryPoint, dist: entryDist})

	for candidates.Len() > 0 {
		// Pop closest candidate
		curr := heap.Pop(candidates).(distNode)

		// Check if we've expanded enough
		if len(result) >= L && curr.dist > result[L-1].dist {
			break
		}

		// Expand neighbors
		for _, neighbor := range b.graph[curr.id] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true

			dist := b.distFunc(b.vectors[neighbor], targetVec)
			heap.Push(candidates, distNode{id: neighbor, dist: dist})

			// Add to result if close enough
			result = append(result, distNode{id: neighbor, dist: dist})
		}

		// Keep result sorted and trimmed
		if len(result) > L*2 {
			sortDistNodes(result)
			result = result[:L]
		}
	}

	// Sort and return top L
	sortDistNodes(result)
	if len(result) > L {
		result = result[:L]
	}

	ids := make([]uint32, len(result))
	for i, r := range result {
		ids[i] = r.id
	}
	return ids
}

// robustPrune implements the Vamana robust pruning algorithm.
func (b *Builder) robustPrune(node uint32, candidates []uint32, R int, alpha float32) []uint32 {
	nodeVec := b.vectors[node]

	// Compute distances to all candidates
	type candidate struct {
		id   uint32
		dist float32
	}
	cands := make([]candidate, 0, len(candidates))
	for _, c := range candidates {
		if c == node {
			continue
		}
		dist := b.distFunc(b.vectors[c], nodeVec)
		cands = append(cands, candidate{id: c, dist: dist})
	}

	// Sort by distance
	for i := 0; i < len(cands); i++ {
		for j := i + 1; j < len(cands); j++ {
			if cands[j].dist < cands[i].dist {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}

	// Greedy selection with diversity
	selected := make([]uint32, 0, R)
	for _, c := range cands {
		if len(selected) >= R {
			break
		}

		// Check if c is diverse enough from already selected
		diverse := true
		for _, s := range selected {
			distCS := b.distFunc(b.vectors[c.id], b.vectors[s])
			if alpha*distCS < c.dist {
				diverse = false
				break
			}
		}

		if diverse {
			selected = append(selected, c.id)
		}
	}

	return selected
}

// addEdge adds an edge from src to dst, pruning if necessary.
func (b *Builder) addEdge(src, dst uint32, R int, alpha float32) {
	// Check if edge already exists
	for _, neighbor := range b.graph[src] {
		if neighbor == dst {
			return
		}
	}

	// Add edge
	b.graph[src] = append(b.graph[src], dst)

	// Prune if over capacity
	if len(b.graph[src]) > R {
		candidates := make([]uint32, len(b.graph[src]))
		copy(candidates, b.graph[src])
		b.graph[src] = b.robustPrune(src, candidates, R, alpha)
	}
}

// writeIndexFiles writes all index files to disk atomically.
// Uses atomic writes (temp file + rename) to prevent corruption on crash.
func (b *Builder) writeIndexFiles() error {
	return persistence.AtomicSaveToDir(b.indexPath, map[string]func(io.Writer) error{
		MetaFilename:    b.writeMetaToWriter,
		GraphFilename:   b.writeGraphToWriter,
		PQCodesFilename: b.writePQCodesToWriter,
		VectorsFilename: b.writeVectorsToWriter,
	})
}

// writeMetaToWriter writes metadata to an io.Writer (for atomic saves).
func (b *Builder) writeMetaToWriter(w io.Writer) error {
	header := FileHeader{
		Magic:        FormatMagic,
		Version:      FormatVersion,
		Flags:        FlagPQEnabled,
		Dimension:    uint32(b.dim),
		Count:        uint64(len(b.vectors)),
		DistanceType: uint32(b.distType),
		R:            uint32(b.opts.R),
		L:            uint32(b.opts.L),
		Alpha:        uint32(b.opts.Alpha * 1000),
		PQSubvectors: uint32(b.opts.PQSubvectors),
		PQCentroids:  uint32(b.opts.PQCentroids),
	}

	if _, err := header.WriteTo(w); err != nil {
		return err
	}

	// Write entry point
	if err := writeUint32ToWriter(w, b.entryPoint); err != nil {
		return err
	}

	// Write PQ codebooks
	return b.writePQCodebooksToWriter(w)
}

// writeMetaFile writes the metadata file (legacy, kept for compatibility).
func (b *Builder) writeMetaFile() error {
	path := filepath.Join(b.indexPath, MetaFilename)
	return persistence.SaveToFile(path, b.writeMetaToWriter)
}

// writePQCodebooksToWriter writes PQ codebooks to an io.Writer.
func (b *Builder) writePQCodebooksToWriter(w io.Writer) error {
	// Get codebooks from PQ
	codebooks := b.pq.Codebooks()

	for m := 0; m < b.opts.PQSubvectors; m++ {
		for k := 0; k < b.opts.PQCentroids; k++ {
			for _, v := range codebooks[m][k] {
				if err := writeFloat32ToWriter(w, v); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// writePQCodebooks writes PQ codebooks to a file (legacy).
func (b *Builder) writePQCodebooks(w *os.File) error {
	return b.writePQCodebooksToWriter(w)
}

// writeGraphToWriter writes the graph to an io.Writer (for atomic saves).
func (b *Builder) writeGraphToWriter(w io.Writer) error {
	for _, neighbors := range b.graph {
		// Write degree
		if err := writeUint32ToWriter(w, uint32(len(neighbors))); err != nil {
			return err
		}
		// Write neighbors
		for _, neighbor := range neighbors {
			if err := writeUint32ToWriter(w, neighbor); err != nil {
				return err
			}
		}
	}
	return nil
}

// writeGraphFile writes the graph file (legacy, kept for compatibility).
func (b *Builder) writeGraphFile() error {
	path := filepath.Join(b.indexPath, GraphFilename)
	return persistence.SaveToFile(path, b.writeGraphToWriter)
}

// writePQCodesToWriter writes PQ codes to an io.Writer (for atomic saves).
func (b *Builder) writePQCodesToWriter(w io.Writer) error {
	for _, codes := range b.pqCodes {
		if _, err := w.Write(codes); err != nil {
			return err
		}
	}
	return nil
}

// writePQCodesFile writes the PQ codes file (legacy, kept for compatibility).
func (b *Builder) writePQCodesFile() error {
	path := filepath.Join(b.indexPath, PQCodesFilename)
	return persistence.SaveToFile(path, b.writePQCodesToWriter)
}

// writeVectorsToWriter writes vectors to an io.Writer (for atomic saves).
func (b *Builder) writeVectorsToWriter(w io.Writer) error {
	for _, vec := range b.vectors {
		for _, v := range vec {
			if err := writeFloat32ToWriter(w, v); err != nil {
				return err
			}
		}
	}
	return nil
}

// writeVectorsFile writes the vectors file (legacy, kept for compatibility).
func (b *Builder) writeVectorsFile() error {
	path := filepath.Join(b.indexPath, VectorsFilename)
	return persistence.SaveToFile(path, b.writeVectorsToWriter)
}

// distNode is a node with distance for heap operations.
type distNode struct {
	id   uint32
	dist float32
}

// distHeap is a min-heap of distNodes.
type distHeap []distNode

func (h distHeap) Len() int           { return len(h) }
func (h distHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h distHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *distHeap) Push(x interface{}) {
	*h = append(*h, x.(distNode))
}

func (h *distHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// sortDistNodes sorts distance nodes by distance (ascending).
func sortDistNodes(nodes []distNode) {
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			if nodes[j].dist < nodes[i].dist {
				nodes[i], nodes[j] = nodes[j], nodes[i]
			}
		}
	}
}

// Helper functions for binary I/O.

// writeUint32ToWriter writes a uint32 to an io.Writer in little-endian format.
func writeUint32ToWriter(w io.Writer, v uint32) error {
	buf := []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
	_, err := w.Write(buf)
	return err
}

// writeFloat32ToWriter writes a float32 to an io.Writer in little-endian format.
func writeFloat32ToWriter(w io.Writer, v float32) error {
	return writeUint32ToWriter(w, math.Float32bits(v))
}

// writeUint32 writes a uint32 to a file (legacy, for backward compatibility).
func writeUint32(f *os.File, v uint32) error {
	return writeUint32ToWriter(f, v)
}

// writeFloat32 writes a float32 to a file (legacy, for backward compatibility).
func writeFloat32(f *os.File, v float32) error {
	return writeFloat32ToWriter(f, v)
}
