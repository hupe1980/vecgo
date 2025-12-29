package diskann

import (
	"container/heap"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"iter"
	"math"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/bits-and-blooms/bitset"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/mmap"
	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/vectorstore"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

// Compile-time interface checks
var (
	_ index.Index              = (*Index)(nil)
	_ index.TransactionalIndex = (*Index)(nil)
)

// Index is a disk-resident approximate nearest neighbor index with incremental update support.
//
// It implements both index.Index and index.TransactionalIndex for full Vecgo integration.
// Supports:
//   - Incremental inserts (new vectors added to Vamana graph)
//   - Soft deletes (deletion bitmap, skip during search)
//   - Updates (delete + insert)
//   - Background compaction (removes deleted vectors, rebuilds graph, re-trains PQ)
//   - Both mutable mode (New) and read-only mode (Open for Builder-created indexes)
type Index struct {
	dim      int
	distType index.DistanceType
	distFunc index.DistanceFunc
	opts     *Options

	// Graph navigation (in RAM)
	// INVARIANT: graphMu protects both graph and incomingEdges
	entryPointAtomic atomic.Uint32
	graph            [][]uint32          // Adjacency lists (mutable)
	incomingEdges    map[uint32][]uint32 // Reverse index for O(1) incoming edge removal (protected by graphMu)
	graphMu          sync.RWMutex

	// PQ for approximate distances (in RAM)
	pq      *quantization.ProductQuantizer
	pqCodes [][]byte // PQ codes per vector

	// Optional BQ codes for search-only prefiltering (in RAM)
	bq      *quantization.BinaryQuantizer
	bqCodes [][]uint64

	fileFlags uint32 // persisted flags from meta header (read-only indexes)

	// Vectors - mutable columnar storage (for New()) or mmap (for Open())
	vectors    vectorstore.Store // Used in mutable mode
	mmapReader *mmap.File        // Used in read-only mode

	// ID management (for TransactionalIndex)
	nextIDAtomic atomic.Uint32
	freeList     []uint32
	freeListMu   sync.Mutex

	// Deletion tracking
	deleted   *bitset.BitSet
	deletedMu sync.RWMutex

	// Compaction tracking (background goroutine lifecycle)
	compacting      atomic.Bool
	stopCompaction  chan struct{}  // Shutdown signal for compaction worker
	compactionWg    sync.WaitGroup // Tracks compaction worker lifecycle
	compactionStats CompactionStats
	compactionMu    sync.RWMutex

	// Performance optimization: pool visited bitsets to reduce allocations
	visitedPool sync.Pool

	// Index path for persistence
	indexPath string

	// Mode flag
	isReadOnly bool
	count      int // For read-only mode, total vector count

	mu sync.RWMutex
}

// CompactionStats tracks compaction statistics.
type CompactionStats struct {
	LastCompactionTime     int64  // Unix timestamp
	TotalCompactions       uint64 // Total number of compactions performed
	VectorsRemovedTotal    uint64 // Total vectors removed across all compactions
	LastVectorsRemoved     uint32 // Vectors removed in last compaction
	LastCompactionDuration int64  // Duration in milliseconds
}

// New creates a new mutable DiskANN index with incremental update support.
// For read-only indexes built with Builder, use Open() instead.
func New(dim int, distType index.DistanceType, indexPath string, opts *Options) (*Index, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	if dim <= 0 {
		return nil, &index.ErrInvalidDimension{Dimension: dim}
	}

	if dim%opts.PQSubvectors != 0 {
		return nil, fmt.Errorf("diskann: dimension %d not divisible by PQSubvectors %d", dim, opts.PQSubvectors)
	}
	if opts.EnableBinaryPrefilter {
		if opts.BinaryPrefilterMaxNormalizedDistance < 0 || opts.BinaryPrefilterMaxNormalizedDistance > 1 {
			return nil, fmt.Errorf("diskann: BinaryPrefilterMaxNormalizedDistance must be in [0, 1], got %f", opts.BinaryPrefilterMaxNormalizedDistance)
		}
	}

	// Create PQ quantizer (will be trained incrementally or from samples)
	pq, err := quantization.NewProductQuantizer(dim, opts.PQSubvectors, opts.PQCentroids)
	if err != nil {
		return nil, fmt.Errorf("diskann: create PQ: %w", err)
	}

	idx := &Index{
		dim:            dim,
		distType:       distType,
		distFunc:       index.NewDistanceFunc(distType),
		opts:           opts,
		pq:             pq,
		graph:          make([][]uint32, 0, 1024),
		incomingEdges:  make(map[uint32][]uint32),
		pqCodes:        make([][]byte, 0, 1024),
		bq:             nil,
		bqCodes:        nil,
		vectors:        columnar.New(dim),
		deleted:        bitset.New(1024),
		freeList:       make([]uint32, 0),
		indexPath:      indexPath,
		isReadOnly:     false,
		stopCompaction: make(chan struct{}),
	}

	if opts.EnableBinaryPrefilter {
		idx.bq = quantization.NewBinaryQuantizer(dim)
		idx.bqCodes = make([][]uint64, 0, 1024)
	}

	idx.nextIDAtomic.Store(0)
	idx.entryPointAtomic.Store(0) // Will be set on first insert

	// Initialize visited pool for performance
	idx.visitedPool = sync.Pool{
		New: func() interface{} {
			return bitset.New(1024)
		},
	}

	// Start background compaction if enabled
	if opts.EnableAutoCompaction {
		idx.compactionWg.Add(1)
		go idx.backgroundCompaction()
	}

	return idx, nil
}

// Open loads a DiskANN index from disk (read-only mode for Builder-created indexes).
// For new mutable indexes, use New() instead.
func Open(indexPath string, opts *Options) (*Index, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	idx := &Index{
		opts:          opts,
		indexPath:     indexPath,
		isReadOnly:    true,
		deleted:       bitset.New(1024),
		freeList:      make([]uint32, 0),
		incomingEdges: make(map[uint32][]uint32),
	}

	// Load metadata (sets dim, count, distType, distFunc, entryPoint, pq)
	if err := idx.loadMeta(indexPath); err != nil {
		return nil, fmt.Errorf("diskann: load meta: %w", err)
	}

	// Load graph
	if err := idx.loadGraph(indexPath); err != nil {
		return nil, fmt.Errorf("diskann: load graph: %w", err)
	}

	// Load PQ codes
	if err := idx.loadPQCodes(indexPath); err != nil {
		return nil, fmt.Errorf("diskann: load pq codes: %w", err)
	}

	// Optional: Load BQ codes for search-only prefiltering
	if opts.EnableBinaryPrefilter {
		if idx.fileFlags&FlagBQEnabled == 0 {
			return nil, fmt.Errorf("diskann: binary prefilter enabled but index has no %s (rebuild with EnableBinaryPrefilter)", BQCodesFilename)
		}
		if err := idx.loadBQCodes(indexPath); err != nil {
			return nil, fmt.Errorf("diskann: load bq codes: %w", err)
		}
		idx.bq = quantization.NewBinaryQuantizer(idx.dim)
	}

	// Mmap vectors file
	vectorsPath := filepath.Join(indexPath, VectorsFilename)
	reader, err := mmap.Open(vectorsPath)
	if err != nil {
		return nil, fmt.Errorf("diskann: mmap vectors: %w", err)
	}
	idx.mmapReader = reader

	// Set nextID for potential future mutations (not used in read-only mode)
	idx.nextIDAtomic.Store(uint32(idx.count))

	// Initialize visited pool for performance
	idx.visitedPool = sync.Pool{
		New: func() interface{} {
			return bitset.New(1024)
		},
	}

	return idx, nil
}

// Close releases resources held by the index gracefully.
//
// This method:
// 1. Signals the background compaction worker to stop (if running)
// 2. Waits for the worker to finish (ensuring clean shutdown)
// 3. Closes mmap resources (if any)
//
// After Close() returns, the index is no longer usable.
func (idx *Index) Close() error {
	idx.mu.Lock()

	// Stop background compaction if running
	if idx.stopCompaction != nil {
		close(idx.stopCompaction)
		idx.mu.Unlock()
		idx.compactionWg.Wait() // Wait for compaction worker to finish (ensures no goroutine leak)
		idx.mu.Lock()
	}

	idx.mu.Unlock()

	// Close mmap resources if present
	if idx.mmapReader != nil {
		return idx.mmapReader.Close()
	}
	return nil
}

// loadMeta loads index metadata and PQ codebooks.
func (idx *Index) loadMeta(indexPath string) error {
	path := filepath.Join(indexPath, MetaFilename)
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Read header
	var header FileHeader
	if _, err := header.ReadFrom(f); err != nil {
		return err
	}
	if err := header.Validate(); err != nil {
		return err
	}

	idx.fileFlags = header.Flags

	idx.dim = int(header.Dimension)
	idx.count = int(header.Count)
	idx.distType = header.DistType()
	idx.distFunc = index.NewDistanceFunc(idx.distType)

	// Restore options from header (override user-provided opts with file values)
	idx.opts.R = int(header.R)
	idx.opts.L = int(header.L)
	idx.opts.Alpha = float32(header.Alpha) / 1000.0
	idx.opts.PQSubvectors = int(header.PQSubvectors)
	idx.opts.PQCentroids = int(header.PQCentroids)

	// Read entry point
	var entryBuf [4]byte
	if _, err := io.ReadFull(f, entryBuf[:]); err != nil {
		return err
	}
	idx.entryPointAtomic.Store(binary.LittleEndian.Uint32(entryBuf[:]))

	// Load PQ codebooks
	return idx.loadPQCodebooks(f, &header)
}

func (idx *Index) loadBQCodes(indexPath string) error {
	path := filepath.Join(indexPath, BQCodesFilename)
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	words := (idx.dim + 63) / 64
	idx.bqCodes = make([][]uint64, idx.count)
	buf := make([]byte, 8)
	for i := 0; i < idx.count; i++ {
		codes := make([]uint64, words)
		for w := 0; w < words; w++ {
			if _, err := io.ReadFull(f, buf); err != nil {
				return err
			}
			codes[w] = binary.LittleEndian.Uint64(buf)
		}
		idx.bqCodes[i] = codes
	}
	return nil
}

func (idx *Index) loadPQCodebooks(r io.Reader, header *FileHeader) error {
	M := int(header.PQSubvectors)
	K := int(header.PQCentroids)
	subDim := idx.dim / M

	// Create PQ with loaded codebooks
	var err error
	idx.pq, err = quantization.NewProductQuantizer(idx.dim, M, K)
	if err != nil {
		return err
	}

	// Read codebooks
	codebooks := make([][][]float32, M)
	for m := 0; m < M; m++ {
		codebooks[m] = make([][]float32, K)
		for k := 0; k < K; k++ {
			codebooks[m][k] = make([]float32, subDim)
			for d := 0; d < subDim; d++ {
				var buf [4]byte
				if _, err := io.ReadFull(r, buf[:]); err != nil {
					return err
				}
				bits := binary.LittleEndian.Uint32(buf[:])
				codebooks[m][k][d] = math.Float32frombits(bits)
			}
		}
	}

	// Set codebooks
	idx.pq.SetCodebooks(codebooks)
	return nil
}

func (idx *Index) loadGraph(indexPath string) error {
	path := filepath.Join(indexPath, GraphFilename)
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	idx.graph = make([][]uint32, idx.count)
	buf := make([]byte, 4)

	for i := 0; i < idx.count; i++ {
		// Read degree
		if _, err := io.ReadFull(f, buf); err != nil {
			return err
		}
		degree := binary.LittleEndian.Uint32(buf)

		// Read neighbors
		idx.graph[i] = make([]uint32, degree)
		for j := uint32(0); j < degree; j++ {
			if _, err := io.ReadFull(f, buf); err != nil {
				return err
			}
			idx.graph[i][j] = binary.LittleEndian.Uint32(buf)
		}
	}

	// Build incoming edges reverse index
	for nodeID := uint32(0); nodeID < uint32(idx.count); nodeID++ {
		for _, neighbor := range idx.graph[nodeID] {
			idx.incomingEdges[neighbor] = append(idx.incomingEdges[neighbor], nodeID)
		}
	}

	return nil
}

func (idx *Index) loadPQCodes(indexPath string) error {
	path := filepath.Join(indexPath, PQCodesFilename)
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	M := idx.opts.PQSubvectors
	idx.pqCodes = make([][]byte, idx.count)

	for i := 0; i < idx.count; i++ {
		idx.pqCodes[i] = make([]byte, M)
		if _, err := io.ReadFull(f, idx.pqCodes[i]); err != nil {
			return err
		}
	}

	return nil
}

// ============================================================================
// TransactionalIndex Implementation (ID Allocation)
// ============================================================================

// AllocateID reserves a new ID for insertion.
func (idx *Index) AllocateID() uint32 {
	idx.freeListMu.Lock()
	defer idx.freeListMu.Unlock()

	if len(idx.freeList) > 0 {
		id := idx.freeList[len(idx.freeList)-1]
		idx.freeList = idx.freeList[:len(idx.freeList)-1]
		return id
	}

	return idx.nextIDAtomic.Add(1) - 1
}

// ReleaseID returns a previously allocated but unused ID.
func (idx *Index) ReleaseID(id uint32) {
	idx.freeListMu.Lock()
	defer idx.freeListMu.Unlock()
	idx.freeList = append(idx.freeList, id)
}

// ============================================================================
// Index Mutations (Insert/Delete/Update)
// ============================================================================

// Insert adds a vector to the index using incremental Vamana construction.
func (idx *Index) Insert(ctx context.Context, v []float32) (uint32, error) {
	if idx.isReadOnly {
		return 0, fmt.Errorf("diskann: cannot insert into read-only index")
	}

	if len(v) != idx.dim {
		return 0, &index.ErrDimensionMismatch{Expected: idx.dim, Actual: len(v)}
	}

	id := idx.AllocateID()
	if err := idx.ApplyInsert(ctx, id, v); err != nil {
		idx.ReleaseID(id)
		return 0, err
	}

	return id, nil
}

// ApplyInsert performs the actual insert (used by transactions and recovery).
func (idx *Index) ApplyInsert(ctx context.Context, id uint32, v []float32) error {
	if len(v) != idx.dim {
		return &index.ErrDimensionMismatch{Expected: idx.dim, Actual: len(v)}
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Store vector
	if err := idx.vectors.SetVector(id, v); err != nil {
		return fmt.Errorf("diskann: store vector: %w", err)
	}

	// Extend graph and pqCodes if needed
	idx.graphMu.Lock()
	for uint32(len(idx.graph)) <= id {
		idx.graph = append(idx.graph, nil)
	}
	for uint32(len(idx.pqCodes)) <= id {
		idx.pqCodes = append(idx.pqCodes, nil)
	}
	if idx.opts.EnableBinaryPrefilter {
		for uint32(len(idx.bqCodes)) <= id {
			idx.bqCodes = append(idx.bqCodes, nil)
		}
		if idx.bq != nil {
			idx.bqCodes[id] = idx.bq.EncodeUint64(v)
		}
	}

	// Compute PQ code
	// NOTE: During early inserts (before PQ training), distances are exact.
	// After compaction, PQ is retrained on live data and distances become approximate.
	// This causes search behavior to shift after compaction, which is expected.
	if idx.pq.IsTrained() {
		idx.pqCodes[id] = idx.pq.Encode(v)
	} else {
		// PQ not trained yet - use empty code (will be rebuilt later)
		idx.pqCodes[id] = make([]byte, idx.opts.PQSubvectors)
	}

	// Connect to graph using greedy search + robust pruning
	neighbors := idx.findNeighborsForInsert(v, id, idx.opts.R, idx.opts.L)
	idx.graph[id] = neighbors

	// Reverse edge repair with re-pruning (MANDATORY for Vamana correctness)
	idx.repairReverseEdges(id, v, neighbors)
	idx.graphMu.Unlock()

	// Update entry point: prefer high-degree nodes for better search starting point
	// This is important for incremental Vamana to maintain good recall
	entryPoint := idx.entryPointAtomic.Load()
	if idx.count == 0 {
		// First insert - set as entry point
		idx.entryPointAtomic.Store(id)
	} else if entryPoint < uint32(len(idx.graph)) {
		// Periodically update entry point to high-degree node
		if len(neighbors) > len(idx.graph[entryPoint]) {
			idx.entryPointAtomic.Store(id)
		}
	}
	idx.count++

	// Clear deletion bit
	idx.deletedMu.Lock()
	idx.deleted.Clear(uint(id))
	idx.deletedMu.Unlock()

	return nil
}

// findNeighborsForInsert finds the best neighbors for a new vector using greedy search.
func (idx *Index) findNeighborsForInsert(v []float32, excludeID uint32, R, L int) []uint32 {
	return idx.findNeighborsForInsertWithGraph(v, excludeID, R, L, idx.graph)
}

// findNeighborsForInsertWithGraph finds the best neighbors using a specified graph.
// This is critical for updates: the graph parameter must be a pre-mutation snapshot,
// otherwise greedy search navigates a partially disconnected graph (Vamana violation).
func (idx *Index) findNeighborsForInsertWithGraph(v []float32, excludeID uint32, R, L int, graph [][]uint32) []uint32 {
	if idx.count == 0 {
		return nil
	}

	// Greedy search from entry point
	entryPoint := idx.entryPointAtomic.Load()
	if entryPoint >= uint32(len(graph)) {
		return nil
	}

	// CRITICAL: Use exact distances for graph construction, not PQ
	// PQ distances drift as vectors are added, corrupting neighbor selection.
	// PQ is only used during search (after graph is built).
	var distTable [][]float32 = nil // Force exact distance computation

	// BFS/greedy search to find L nearest candidates
	// Use bitset for visited tracking (memory efficient, faster than map at scale)
	maxID := uint32(len(graph))
	visited := idx.getVisitedBitset(uint(maxID))
	defer idx.putVisitedBitset(visited)

	candidates := &distHeap{}
	heap.Init(candidates)

	entryDist := idx.computeDistance(v, entryPoint, distTable)
	heap.Push(candidates, distNode{id: entryPoint, dist: entryDist})
	visited.Set(uint(entryPoint))

	results := make([]distNode, 0, L)
	results = append(results, distNode{id: entryPoint, dist: entryDist})

	// Expansion limit: explore up to L*3 candidates for good quality without unbounded search
	expansionLimit := L * 3
	if expansionLimit < 300 {
		expansionLimit = 300
	}

	for candidates.Len() > 0 && len(results) < expansionLimit {
		curr := heap.Pop(candidates).(distNode)

		// Quality termination: stop when current candidate is worse than L-th best
		if len(results) >= L {
			sortDistNodes(results)
			if curr.dist > results[L-1].dist {
				break
			}
		}

		if curr.id < uint32(len(graph)) && graph[curr.id] != nil {
			for _, neighbor := range graph[curr.id] {
				if neighbor >= maxID || visited.Test(uint(neighbor)) || neighbor == excludeID {
					continue
				}
				visited.Set(uint(neighbor))

				// Skip deleted vectors
				idx.deletedMu.RLock()
				isDeleted := idx.deleted.Test(uint(neighbor))
				idx.deletedMu.RUnlock()
				if isDeleted {
					continue
				}

				dist := idx.computeDistance(v, neighbor, distTable)
				heap.Push(candidates, distNode{id: neighbor, dist: dist})
				results = append(results, distNode{id: neighbor, dist: dist})
			}
		}
	}

	// Sort by distance and apply robust pruning
	sortDistNodes(results)
	return idx.robustPrune(v, results, R)
}

// robustPrune applies the Vamana pruning strategy to select diverse neighbors.
// Uses exact distances for correct α-RNG geometry during graph construction.
func (idx *Index) robustPrune(center []float32, candidates []distNode, R int) []uint32 {
	if len(candidates) == 0 {
		return nil
	}

	alpha := idx.opts.Alpha
	result := make([]uint32, 0, R)

	for _, cand := range candidates {
		if len(result) >= R {
			break
		}

		// Check if this candidate is dominated by existing neighbors
		// α-RNG test: reject cand if ∃ selected s.t. α * d(cand, selected) < d(cand, center)
		dominated := false

		// Fetch candidate vector ONCE per candidate
		candVec := idx.getVector(cand.id)
		if candVec == nil {
			continue
		}

		// Test against all already-selected neighbors (exact distance, no PQ)
		for _, selected := range result {
			distCandSelected := idx.computeDistance(candVec, selected, nil)
			if float32(alpha)*distCandSelected < cand.dist {
				dominated = true
				break
			}
		}

		if !dominated {
			result = append(result, cand.id)
		}
	}

	return result
}

// repairReverseEdges performs reverse-edge repair with re-pruning.
// This is MANDATORY for Vamana correctness after any node insertion/update.
// For each neighbor v of node id:
//  1. Add id as a candidate to v's adjacency list
//  2. Re-prune v's list to maintain α-RNG property
//  3. Track incoming edges for O(1) removal during updates
func (idx *Index) repairReverseEdges(id uint32, v []float32, neighbors []uint32) {
	for _, neighbor := range neighbors {
		if neighbor >= uint32(len(idx.graph)) {
			continue
		}

		// Get neighbor's vector for distance computation
		neighborVec := idx.getVector(neighbor)
		if neighborVec == nil {
			continue
		}

		// Collect current neighbors as candidates (using exact distances)
		candidates := make([]distNode, 0, len(idx.graph[neighbor])+1)
		for _, n := range idx.graph[neighbor] {
			nVec := idx.getVector(n)
			if nVec == nil {
				continue
			}
			dist := idx.distFunc(neighborVec, nVec)
			candidates = append(candidates, distNode{id: n, dist: dist})
		}

		// Add the new/updated node as a candidate
		distToNew := idx.distFunc(neighborVec, v)
		candidates = append(candidates, distNode{id: id, dist: distToNew})

		// Sort and re-prune to maintain Vamana invariants
		sortDistNodes(candidates)
		oldNeighbors := idx.graph[neighbor]
		newNeighbors := idx.robustPrune(neighborVec, candidates, idx.opts.R)
		idx.graph[neighbor] = newNeighbors

		// Track incoming edges: if id is now in neighbor's list, record it
		for _, n := range newNeighbors {
			if n == id {
				idx.incomingEdges[id] = append(idx.incomingEdges[id], neighbor)
				break
			}
		}

		// Remove from incoming edges if id was removed during pruning
		wasInOld := false
		isInNew := false
		for _, n := range oldNeighbors {
			if n == id {
				wasInOld = true
				break
			}
		}
		for _, n := range newNeighbors {
			if n == id {
				isInNew = true
				break
			}
		}
		if wasInOld && !isInNew {
			// Remove neighbor from incoming edges list
			filtered := idx.incomingEdges[id][:0]
			for _, inc := range idx.incomingEdges[id] {
				if inc != neighbor {
					filtered = append(filtered, inc)
				}
			}
			idx.incomingEdges[id] = filtered
		}
	}
}

// computeDistance computes distance using PQ (if trained) or exact distance.
func (idx *Index) computeDistance(v []float32, id uint32, distTable [][]float32) float32 {
	if distTable != nil && id < uint32(len(idx.pqCodes)) && idx.pqCodes[id] != nil {
		return idx.pqDistance(distTable, idx.pqCodes[id])
	}
	// Fall back to exact distance
	vec := idx.getVector(id)
	if vec == nil {
		return math.MaxFloat32
	}
	return idx.distFunc(v, vec)
}

// BatchInsert adds multiple vectors in a single operation.
func (idx *Index) BatchInsert(ctx context.Context, vectors [][]float32) index.BatchInsertResult {
	result := index.BatchInsertResult{
		IDs:    make([]uint32, len(vectors)),
		Errors: make([]error, len(vectors)),
	}

	for i, v := range vectors {
		id, err := idx.Insert(ctx, v)
		result.IDs[i] = id
		result.Errors[i] = err
	}

	return result
}

// Delete removes a vector from the index using soft delete.
func (idx *Index) Delete(ctx context.Context, id uint32) error {
	if idx.isReadOnly {
		return fmt.Errorf("diskann: cannot delete from read-only index")
	}
	return idx.ApplyDelete(ctx, id)
}

// ApplyDelete performs the actual delete (used by transactions and recovery).
func (idx *Index) ApplyDelete(ctx context.Context, id uint32) error {
	idx.deletedMu.Lock()
	defer idx.deletedMu.Unlock()

	if id >= uint32(idx.count) {
		return &index.ErrNodeNotFound{ID: id}
	}

	idx.deleted.Set(uint(id))
	return nil
}

// Update updates a vector in the index.
func (idx *Index) Update(ctx context.Context, id uint32, v []float32) error {
	if idx.isReadOnly {
		return fmt.Errorf("diskann: cannot update read-only index")
	}
	return idx.ApplyUpdate(ctx, id, v)
}

// ApplyUpdate performs the actual update (used by transactions and recovery).
func (idx *Index) ApplyUpdate(ctx context.Context, id uint32, v []float32) error {
	// Soft delete old, then insert with same ID
	if err := idx.ApplyDelete(ctx, id); err != nil {
		return err
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Update vector
	if err := idx.vectors.SetVector(id, v); err != nil {
		return fmt.Errorf("diskann: update vector: %w", err)
	}

	// NOTE: PQ code will be recomputed on compaction.
	// During incremental updates, graph uses exact distances.
	// This ensures graph quality until compaction retrains PQ.

	idx.graphMu.Lock()
	defer idx.graphMu.Unlock()

	// CRITICAL: Snapshot graph BEFORE any mutations
	// findNeighborsForInsert must search a fully connected graph,
	// otherwise greedy search explores damaged neighborhoods (Vamana violation)
	snapshot := make([][]uint32, len(idx.graph))
	for i := range idx.graph {
		if idx.graph[i] != nil {
			snapshot[i] = append([]uint32(nil), idx.graph[i]...)
		}
	}

	// Remove incoming edges using reverse index (O(R) instead of O(N*R))
	for _, incoming := range idx.incomingEdges[id] {
		if incoming >= uint32(len(idx.graph)) {
			continue
		}
		filtered := idx.graph[incoming][:0]
		for _, n := range idx.graph[incoming] {
			if n != id {
				filtered = append(filtered, n)
			}
		}
		idx.graph[incoming] = filtered
	}
	// Clear incoming edges for this node
	idx.incomingEdges[id] = nil

	// Rebuild outgoing edges using PRE-MUTATION snapshot
	// NOTE: Use adaptive L (1.5x higher) during updates to compensate for slightly
	// weaker exploration from termination condition (len(results) < L*2).
	// This maintains update quality close to fresh inserts.
	adaptiveL := idx.opts.L + idx.opts.L/2
	neighbors := idx.findNeighborsForInsertWithGraph(v, id, idx.opts.R, adaptiveL, snapshot)
	idx.graph[id] = neighbors

	// MANDATORY: Perform reverse-edge repair (same as insert)
	idx.repairReverseEdges(id, v, neighbors)

	// Clear deletion bit AFTER graph is fully repaired
	idx.deletedMu.Lock()
	idx.deleted.Clear(uint(id))
	idx.deletedMu.Unlock()

	return nil
}

// VectorByID retrieves a vector by its ID.
func (idx *Index) VectorByID(ctx context.Context, id uint32) ([]float32, error) {
	idx.deletedMu.RLock()
	isDeleted := idx.deleted.Test(uint(id))
	idx.deletedMu.RUnlock()

	if isDeleted {
		return nil, &index.ErrNodeDeleted{ID: id}
	}

	vec := idx.getVector(id)
	if vec == nil {
		return nil, &index.ErrNodeNotFound{ID: id}
	}

	return vec, nil
}

// getVector retrieves a vector from either columnar store or mmap.
func (idx *Index) getVector(id uint32) []float32 {
	if idx.isReadOnly {
		return idx.readVectorMmap(id)
	}

	vec, ok := idx.vectors.GetVector(id)
	if !ok {
		return nil
	}
	return vec
}

// readVectorMmap reads a vector from the mmap'd file.
func (idx *Index) readVectorMmap(id uint32) []float32 {
	if idx.mmapReader == nil {
		return nil
	}

	offset := int(id) * int(idx.dim) * 4
	if offset < 0 || offset+int(idx.dim)*4 > len(idx.mmapReader.Data) {
		return nil
	}

	// Zero-copy access
	// Note: This assumes LittleEndian architecture and 4-byte alignment.
	// mmap data is usually page-aligned.
	// If offset is not aligned, we must copy.
	if offset%4 != 0 {
		vec := make([]float32, idx.dim)
		buf := idx.mmapReader.Data[offset : offset+int(idx.dim)*4]
		for i := 0; i < idx.dim; i++ {
			bits := binary.LittleEndian.Uint32(buf[i*4:])
			vec[i] = math.Float32frombits(bits)
		}
		return vec
	}

	return unsafe.Slice((*float32)(unsafe.Pointer(&idx.mmapReader.Data[offset])), idx.dim)
}

// ============================================================================
// Search Operations
// ============================================================================

// KNNSearch performs k-nearest neighbor search with optional pre-filtering.
// filter (in opts): Applied DURING graph traversal for correct recall and performance.
func (idx *Index) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	if len(query) != idx.dim {
		return nil, &index.ErrDimensionMismatch{Expected: idx.dim, Actual: len(query)}
	}
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	idx.mu.RLock()
	count := idx.count
	idx.mu.RUnlock()

	if count == 0 {
		return nil, nil
	}

	// Build PQ distance table for query
	var distTable [][]float32
	if idx.pq.IsTrained() {
		distTable = idx.pq.BuildDistanceTable(query)
	}

	// Extract filter for pre-filtering during graph search
	var filter func(uint32) bool
	if opts != nil {
		filter = opts.Filter
	}

	// Optional BQ query encoding (search-only prefilter)
	var queryBQ []uint64
	if idx.opts.EnableBinaryPrefilter && idx.bq != nil && len(idx.bqCodes) > 0 {
		queryBQ = idx.bq.EncodeUint64(query)
	}

	// Phase 1: Graph search using PQ distances WITH PRE-FILTERING
	rerankK := idx.opts.RerankK
	if rerankK < k {
		rerankK = k * 2
	}
	candidates := idx.beamSearch(query, distTable, queryBQ, rerankK, filter)

	// Phase 2: Rerank using exact distances (no post-filtering needed)
	results := idx.rerank(query, candidates, k, nil) // Filter already applied in beamSearch

	return results, nil
}

// beamSearch performs beam search through the graph using PQ distances with optional pre-filtering.
// filter: if not nil, nodes are filtered DURING graph traversal (not after).
// This ensures correct recall and reduces wasted distance computations.
// Uses proper DiskANN termination: stop when current candidate is worse than worst in beam.
func (idx *Index) beamSearch(query []float32, distTable [][]float32, queryBQ []uint64, topK int, filter func(uint32) bool) []distNode {
	idx.graphMu.RLock()
	defer idx.graphMu.RUnlock()

	if len(idx.graph) == 0 {
		return nil
	}

	// Min-heap for candidates
	candidates := &distHeap{}
	heap.Init(candidates)

	// Start from entry point
	// NOTE: If entry point is deleted, it's still used for graph navigation
	// (deleted nodes are skipped when adding to results). Compaction resets
	// entry point to a live node. This matches DiskANN behavior.
	entryPoint := idx.entryPointAtomic.Load()
	if entryPoint >= uint32(len(idx.graph)) {
		return nil
	}

	entryDist := idx.computeDistance(query, entryPoint, distTable)
	heap.Push(candidates, distNode{id: entryPoint, dist: entryDist})

	// Use bitset for visited tracking (memory efficient, faster than map at scale)
	maxID := uint32(len(idx.graph))
	visited := idx.getVisitedBitset(uint(maxID))
	defer idx.putVisitedBitset(visited)
	visited.Set(uint(entryPoint))

	// Result list (sorted by distance, worst at end)
	results := make([]distNode, 0, topK*2)

	// Skip deleted entry point (and apply filter)
	idx.deletedMu.RLock()
	isDeleted := idx.deleted.Test(uint(entryPoint))
	idx.deletedMu.RUnlock()

	// PRE-FILTER: Only add to results if passes filter
	if !isDeleted && (filter == nil || filter(entryPoint)) {
		results = append(results, distNode{id: entryPoint, dist: entryDist})
	}

	// DiskANN beam search: expand until no better candidates exist
	for candidates.Len() > 0 {
		// Pop closest candidate
		curr := heap.Pop(candidates).(distNode)

		// CRITICAL FIX: Proper DiskANN termination condition
		// If beam is full and current candidate is worse than worst in beam, stop
		if len(results) >= topK {
			worstInBeam := results[len(results)-1].dist
			if curr.dist > worstInBeam {
				break
			}
		}

		// Expand neighbors
		if curr.id < uint32(len(idx.graph)) && idx.graph[curr.id] != nil {
			for _, neighbor := range idx.graph[curr.id] {
				if neighbor >= maxID || visited.Test(uint(neighbor)) {
					continue
				}
				visited.Set(uint(neighbor))

				// Skip deleted vectors
				idx.deletedMu.RLock()
				isDeleted := idx.deleted.Test(uint(neighbor))
				idx.deletedMu.RUnlock()

				// PRE-FILTER: Skip filtered nodes BEFORE computing distance
				if filter != nil && !filter(neighbor) {
					continue
				}

				// Optional BQ prefilter: skip nodes that are too far in Hamming space
				if queryBQ != nil && neighbor < uint32(len(idx.bqCodes)) && idx.bqCodes[neighbor] != nil {
					norm := quantization.NormalizedHammingDistance(queryBQ, idx.bqCodes[neighbor], idx.dim)
					if norm > idx.opts.BinaryPrefilterMaxNormalizedDistance {
						continue
					}
				}

				dist := idx.computeDistance(query, neighbor, distTable)
				heap.Push(candidates, distNode{id: neighbor, dist: dist})

				if !isDeleted {
					results = append(results, distNode{id: neighbor, dist: dist})
					// Keep results sorted for termination check
					sortDistNodes(results)
					if len(results) > topK*2 {
						results = results[:topK*2]
					}
				}
			}
		}
	}

	// Sort and return top candidates
	sortDistNodes(results)
	if len(results) > topK {
		results = results[:topK]
	}

	return results
}

// pqDistance computes approximate distance using PQ codes.
func (idx *Index) pqDistance(distTable [][]float32, codes []byte) float32 {
	var dist float32
	for m, code := range codes {
		dist += distTable[m][code]
	}
	return dist
}

// rerank fetches vectors and computes exact distances.
// filter: Optional post-filtering (but should be nil since pre-filtering is done in beamSearch).
// Kept for backward compatibility, but pre-filtering is the recommended approach.
func (idx *Index) rerank(query []float32, candidates []distNode, k int, filter func(uint32) bool) []index.SearchResult {
	// Fetch vectors and compute exact distances
	results := make([]distNode, 0, len(candidates))
	for _, c := range candidates {
		// Optional post-filter (should be nil if pre-filtering was used)
		if filter != nil && !filter(c.id) {
			continue
		}

		vec := idx.getVector(c.id)
		if vec == nil {
			continue
		}

		dist := idx.distFunc(query, vec)
		results = append(results, distNode{id: c.id, dist: dist})
	}

	// Sort by exact distance
	sortDistNodes(results)
	if len(results) > k {
		results = results[:k]
	}

	// Convert to SearchResult
	out := make([]index.SearchResult, len(results))
	for i, r := range results {
		out[i] = index.SearchResult{ID: r.id, Distance: r.dist}
	}

	return out
}

// KNNSearchStream returns an iterator over search results.
func (idx *Index) KNNSearchStream(ctx context.Context, query []float32, k int, opts *index.SearchOptions) iter.Seq2[index.SearchResult, error] {
	return func(yield func(index.SearchResult, error) bool) {
		results, err := idx.KNNSearch(ctx, query, k, opts)
		if err != nil {
			yield(index.SearchResult{}, err)
			return
		}

		for _, r := range results {
			if !yield(r, nil) {
				return
			}
		}
	}
}

// BruteSearch performs exact search by scanning all vectors.
func (idx *Index) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	if len(query) != idx.dim {
		return nil, &index.ErrDimensionMismatch{Expected: idx.dim, Actual: len(query)}
	}
	if k <= 0 {
		return nil, index.ErrInvalidK
	}

	idx.mu.RLock()
	count := idx.count
	idx.mu.RUnlock()

	// Max-heap for top k
	h := &maxDistHeap{}
	heap.Init(h)

	for id := uint32(0); id < uint32(count); id++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		// Skip deleted
		idx.deletedMu.RLock()
		isDeleted := idx.deleted.Test(uint(id))
		idx.deletedMu.RUnlock()
		if isDeleted {
			continue
		}

		if filter != nil && !filter(id) {
			continue
		}

		vec := idx.getVector(id)
		if vec == nil {
			continue
		}

		dist := idx.distFunc(query, vec)

		if h.Len() < k {
			heap.Push(h, distNode{id: id, dist: dist})
		} else if dist < (*h)[0].dist {
			heap.Pop(h)
			heap.Push(h, distNode{id: id, dist: dist})
		}
	}

	// Extract results
	results := make([]index.SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		node := heap.Pop(h).(distNode)
		results[i] = index.SearchResult{ID: node.id, Distance: node.dist}
	}

	return results, nil
}

// ============================================================================
// Statistics and Info
// ============================================================================

// Stats returns index statistics.
func (idx *Index) Stats() index.Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	idx.graphMu.RLock()
	totalEdges := 0
	for _, neighbors := range idx.graph {
		totalEdges += len(neighbors)
	}
	idx.graphMu.RUnlock()

	idx.deletedMu.RLock()
	deletedCount := idx.deleted.Count()
	idx.deletedMu.RUnlock()

	liveCount := idx.count - int(deletedCount)
	avgDegree := 0
	if liveCount > 0 {
		avgDegree = totalEdges / liveCount
	}

	mode := "mutable"
	if idx.isReadOnly {
		mode = "read-only"
	}

	storage := map[string]string{
		"VectorCount":   fmt.Sprintf("%d", liveCount),
		"TotalEdges":    fmt.Sprintf("%d", totalEdges),
		"AverageDegree": fmt.Sprintf("%d", avgDegree),
	}

	// Include deletion stats if there are any deleted vectors
	if deletedCount > 0 {
		storage["TotalCount"] = fmt.Sprintf("%d", idx.count)
		storage["DeletedCount"] = fmt.Sprintf("%d", deletedCount)
	}

	return index.Stats{
		Options: map[string]string{
			"Dimension":    fmt.Sprintf("%d", idx.dim),
			"DistanceType": idx.distType.String(),
			"Mode":         mode,
		},
		Parameters: map[string]string{
			"R":            fmt.Sprintf("%d", idx.opts.R),
			"L":            fmt.Sprintf("%d", idx.opts.L),
			"Alpha":        fmt.Sprintf("%.2f", idx.opts.Alpha),
			"PQSubvectors": fmt.Sprintf("%d", idx.opts.PQSubvectors),
			"PQCentroids":  fmt.Sprintf("%d", idx.opts.PQCentroids),
		},
		Storage: storage,
	}
}

// Dimension returns the vector dimension.
func (idx *Index) Dimension() int {
	return idx.dim
}

// Count returns the number of live (non-deleted) vectors.
func (idx *Index) Count() int {
	idx.deletedMu.RLock()
	deletedCount := idx.deleted.Count()
	idx.deletedMu.RUnlock()

	return idx.count - int(deletedCount)
}

// ============================================================================
// Compaction (removes deleted vectors, rebuilds graph, re-trains PQ)
// ============================================================================

// Compact removes all deleted vectors and rebuilds the index for optimal performance.
// This is a blocking operation that acquires exclusive locks.
func (idx *Index) Compact(ctx context.Context) error {
	if idx.isReadOnly {
		return fmt.Errorf("diskann: cannot compact read-only index")
	}

	// Prevent concurrent compactions
	if !idx.compacting.CompareAndSwap(false, true) {
		return fmt.Errorf("diskann: compaction already in progress")
	}
	defer idx.compacting.Store(false)

	startTime := nowMillis()

	idx.deletedMu.RLock()
	deletedCount := idx.deleted.Count()
	idx.deletedMu.RUnlock()

	if deletedCount == 0 {
		return nil // Nothing to compact
	}

	// Acquire exclusive lock for compaction
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Build ID remapping (old ID -> new ID)
	idMap := make(map[uint32]uint32)
	newID := uint32(0)
	liveVectors := make([][]float32, 0, idx.count-int(deletedCount))

	// Collect all live vectors
	for oldID := uint32(0); oldID < uint32(idx.count); oldID++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		idx.deletedMu.RLock()
		isDeleted := idx.deleted.Test(uint(oldID))
		idx.deletedMu.RUnlock()

		if isDeleted {
			continue
		}

		vec := idx.getVector(oldID)
		if vec == nil {
			continue
		}

		// Copy vector
		vecCopy := make([]float32, len(vec))
		copy(vecCopy, vec)
		liveVectors = append(liveVectors, vecCopy)

		idMap[oldID] = newID
		newID++
	}

	if len(liveVectors) == 0 {
		return fmt.Errorf("diskann: no live vectors after compaction")
	}

	// Re-train PQ on live vectors
	if err := idx.pq.Train(liveVectors); err != nil {
		return fmt.Errorf("diskann: re-train PQ: %w", err)
	}

	// Rebuild vectors and PQ codes
	newVectors := columnar.New(idx.dim)
	newPQCodes := make([][]byte, len(liveVectors))
	var newBQCodes [][]uint64
	if idx.opts.EnableBinaryPrefilter && idx.bq != nil {
		newBQCodes = make([][]uint64, len(liveVectors))
	}
	for i, vec := range liveVectors {
		if err := newVectors.SetVector(uint32(i), vec); err != nil {
			return fmt.Errorf("diskann: set vector %d: %w", i, err)
		}
		newPQCodes[i] = idx.pq.Encode(vec)
		if newBQCodes != nil {
			newBQCodes[i] = idx.bq.EncodeUint64(vec)
		}
	}

	// Rebuild graph with remapped IDs
	idx.graphMu.Lock()
	newGraph := make([][]uint32, len(liveVectors))
	for oldID, newID := range idMap {
		if int(oldID) >= len(idx.graph) {
			continue
		}

		oldNeighbors := idx.graph[oldID]
		newNeighbors := make([]uint32, 0, len(oldNeighbors))

		for _, oldNeighbor := range oldNeighbors {
			if newNeighbor, ok := idMap[oldNeighbor]; ok {
				newNeighbors = append(newNeighbors, newNeighbor)
			}
		}

		newGraph[newID] = newNeighbors
	}

	// Update entry point
	oldEntry := idx.entryPointAtomic.Load()
	if newEntry, ok := idMap[oldEntry]; ok {
		idx.entryPointAtomic.Store(newEntry)
	} else if len(liveVectors) > 0 {
		idx.entryPointAtomic.Store(0) // Default to first vector
	}

	// Replace with new structures
	idx.graph = newGraph

	// Rebuild incoming edges reverse index
	idx.incomingEdges = make(map[uint32][]uint32)
	for nodeID := uint32(0); nodeID < uint32(len(newGraph)); nodeID++ {
		for _, neighbor := range newGraph[nodeID] {
			idx.incomingEdges[neighbor] = append(idx.incomingEdges[neighbor], nodeID)
		}
	}

	// Replace other search-critical structures under graphMu to avoid races with concurrent searches.
	idx.pqCodes = newPQCodes
	if newBQCodes != nil {
		idx.bqCodes = newBQCodes
	}
	idx.vectors = newVectors
	idx.count = len(liveVectors)

	idx.graphMu.Unlock()

	// Reset deletion tracking
	idx.deletedMu.Lock()
	idx.deleted = bitset.New(uint(len(liveVectors)))
	idx.deletedMu.Unlock()

	// Reset ID allocation
	idx.nextIDAtomic.Store(uint32(len(liveVectors)))
	idx.freeListMu.Lock()
	idx.freeList = make([]uint32, 0)
	idx.freeListMu.Unlock()

	// Update stats
	duration := nowMillis() - startTime
	idx.compactionMu.Lock()
	idx.compactionStats.LastCompactionTime = startTime
	idx.compactionStats.TotalCompactions++
	idx.compactionStats.VectorsRemovedTotal += uint64(deletedCount)
	idx.compactionStats.LastVectorsRemoved = uint32(deletedCount)
	idx.compactionStats.LastCompactionDuration = duration
	idx.compactionMu.Unlock()

	return nil
}

// ShouldCompact returns true if compaction is recommended based on threshold.
func (idx *Index) ShouldCompact() bool {
	if idx.isReadOnly {
		return false
	}

	idx.deletedMu.RLock()
	deletedCount := idx.deleted.Count()
	idx.deletedMu.RUnlock()

	if idx.count < idx.opts.CompactionMinVectors {
		return false
	}

	deletedRatio := float32(deletedCount) / float32(idx.count)
	return deletedRatio >= idx.opts.CompactionThreshold
}

// CompactionStats returns current compaction statistics.
func (idx *Index) CompactionStats() CompactionStats {
	idx.compactionMu.RLock()
	defer idx.compactionMu.RUnlock()
	return idx.compactionStats
}

// backgroundCompaction runs periodic compaction checks.
// This method runs in a background goroutine and is tracked by compactionWg.
func (idx *Index) backgroundCompaction() {
	defer idx.compactionWg.Done() // Ensure WaitGroup is decremented on exit

	ticker := newTicker(idx.opts.CompactionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if idx.ShouldCompact() {
				ctx := context.Background()
				_ = idx.Compact(ctx) // Ignore errors in background
			}
		case <-idx.stopCompaction:
			return
		}
	}
}

// ============================================================================
// Heap types for search (maxDistHeap only - distHeap/distNode in builder.go)
// ============================================================================

// maxDistHeap is a max-heap of distNodes for top-k selection.
type maxDistHeap []distNode

func (h maxDistHeap) Len() int           { return len(h) }
func (h maxDistHeap) Less(i, j int) bool { return h[i].dist > h[j].dist } // Max heap
func (h maxDistHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *maxDistHeap) Push(x interface{}) {
	*h = append(*h, x.(distNode))
}

func (h *maxDistHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// ============================================================================
// Helper functions
// ============================================================================

// nowMillis returns current time in milliseconds since Unix epoch.
func nowMillis() int64 {
	return time.Now().UnixMilli()
}

// newTicker creates a ticker that ticks every interval seconds.
func newTicker(intervalSeconds int) *time.Ticker {
	return time.NewTicker(time.Duration(intervalSeconds) * time.Second)
}

// getVisitedBitset gets a bitset from the pool, sized appropriately.
func (idx *Index) getVisitedBitset(size uint) *bitset.BitSet {
	bs := idx.visitedPool.Get().(*bitset.BitSet)
	bs.ClearAll()
	if bs.Len() < size {
		// Grow bitset if needed
		bs = bitset.New(size)
	}
	return bs
}

// putVisitedBitset returns a bitset to the pool for reuse.
func (idx *Index) putVisitedBitset(bs *bitset.BitSet) {
	if bs != nil {
		idx.visitedPool.Put(bs)
	}
}
