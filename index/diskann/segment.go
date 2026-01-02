package diskann

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"unsafe"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/mmap"
	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/searcher"
)

// Segment represents an immutable, disk-resident DiskANN index segment.
// It supports concurrent searches but no mutations.
type Segment struct {
	baseID    core.LocalID // The starting Global ID for this segment
	dim       int
	count     uint32
	distType  index.DistanceType
	distFunc  index.DistanceFunc
	opts      *Options
	fileFlags uint32

	// Graph navigation (in RAM)
	entryPoint uint32
	graph      [][]uint32 // Adjacency lists (immutable)

	// PQ for approximate distances (in RAM)
	pq      *quantization.ProductQuantizer
	pqCodes []byte // PQ codes per vector (flat array: count * M)

	// Optional BQ codes for search-only prefiltering (in RAM)
	bq      *quantization.BinaryQuantizer
	bqCodes []uint64 // BQ codes per vector

	// Vectors - mmap (immutable)
	mmapReader *mmap.File

	// Performance optimization: pool visited bitsets
	visitedPool sync.Pool

	path string
}

// OpenSegment opens an existing DiskANN segment from disk.
func OpenSegment(path string, baseID core.LocalID, opts *Options) (*Segment, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	seg := &Segment{
		path:   path,
		opts:   opts,
		baseID: baseID,
	}

	// Load metadata
	if err := seg.loadMeta(path); err != nil {
		return nil, fmt.Errorf("diskann: load meta: %w", err)
	}

	// Load graph
	if err := seg.loadGraph(path); err != nil {
		return nil, fmt.Errorf("diskann: load graph: %w", err)
	}

	// Load PQ codes
	if err := seg.loadPQCodes(path); err != nil {
		return nil, fmt.Errorf("diskann: load pq codes: %w", err)
	}

	// Optional: Load BQ codes
	if opts.EnableBinaryPrefilter {
		if seg.fileFlags&FlagBQEnabled == 0 {
			return nil, fmt.Errorf("diskann: binary prefilter enabled but segment has no %s", BQCodesFilename)
		}
		if err := seg.loadBQCodes(path); err != nil {
			return nil, fmt.Errorf("diskann: load bq codes: %w", err)
		}
		seg.bq = quantization.NewBinaryQuantizer(seg.dim)
	}

	// Mmap vectors file
	vectorsPath := filepath.Join(path, VectorsFilename)
	reader, err := mmap.Open(vectorsPath)
	if err != nil {
		return nil, fmt.Errorf("diskann: mmap vectors: %w", err)
	}
	seg.mmapReader = reader

	// Initialize visited pool
	seg.visitedPool = sync.Pool{
		New: func() interface{} {
			return bitset.New(1024)
		},
	}

	return seg, nil
}

// Close releases resources held by the segment.
func (s *Segment) Close() error {
	if s.mmapReader != nil {
		return s.mmapReader.Close()
	}
	return nil
}

// Search performs k-nearest neighbor search on the segment.
func (s *Segment) Search(query []float32, k int, filter func(core.LocalID) bool) ([]index.SearchResult, error) {
	if len(query) != s.dim {
		return nil, &index.ErrDimensionMismatch{Expected: s.dim, Actual: len(query)}
	}

	// Build PQ distance table
	var distTable []float32
	if s.pq != nil {
		distTable = s.pq.BuildDistanceTable(query)
	}

	// Optional BQ query encoding
	var queryBQ []uint64
	if s.opts.EnableBinaryPrefilter && s.bq != nil && len(s.bqCodes) > 0 {
		queryBQ = s.bq.EncodeUint64(query)
	}

	// Adapt filter to internal IDs
	internalFilter := func(internalID uint32) bool {
		if filter == nil {
			return true
		}
		return filter(core.LocalID(internalID) + s.baseID)
	}

	// Phase 1: Beam search
	rerankK := s.opts.RerankK
	if rerankK < k {
		rerankK = k * 2
	}

	// Create a temporary scratch for standalone search
	scratch := &searchScratch{
		candidates: make(candidateHeap, 0, 128),
		visited:    bitset.New(1024),
		beamBuf:    make([]distNode, 0, 256),
	}

	candidates := s.beamSearch(query, distTable, queryBQ, rerankK, internalFilter, scratch)

	// Phase 2: Rerank
	results := s.rerank(query, candidates, k)

	// Map to Global IDs
	for i := range results {
		results[i].ID += uint32(s.baseID)
	}

	return results, nil
}

// SearchWithBuffer performs KNN search and appends results to the provided buffer.
func (s *Segment) SearchWithBuffer(ctx context.Context, query []float32, k int, distTable []float32, filter func(core.LocalID) bool, scratch *searchScratch, buf *[]index.SearchResult) error {
	// Local filter wrapper to handle Global ID -> Local ID mapping
	var localFilter func(uint32) bool
	if filter != nil {
		localFilter = func(localID uint32) bool {
			return filter(s.baseID + core.LocalID(localID))
		}
	}

	// Phase 1: Beam Search
	rerankK := s.opts.RerankK
	if rerankK < k {
		rerankK = k * 2
	}

	var queryBQ []uint64
	if s.opts.EnableBinaryPrefilter && s.bq != nil && len(s.bqCodes) > 0 {
		queryBQ = s.bq.EncodeUint64(query)
	}

	// Build PQ distance table if not provided and PQ is available
	if distTable == nil && s.pq != nil {
		distTable = s.pq.BuildDistanceTable(query)
	}

	candidates := s.beamSearch(query, distTable, queryBQ, rerankK, localFilter, scratch)

	// Phase 2: Rerank and append to buffer
	// We inline rerank logic to append directly
	startLen := len(*buf)
	for _, c := range candidates {
		vec := s.getVector(c.id)
		if vec == nil {
			continue
		}
		dist := s.distFunc(query, vec)
		*buf = append(*buf, index.SearchResult{ID: uint32(s.baseID) + c.id, Distance: dist})
	}

	// Sort the appended part
	newResults := (*buf)[startLen:]
	sort.Slice(newResults, func(i, j int) bool {
		return newResults[i].Distance < newResults[j].Distance
	})

	// Trim to k
	if len(newResults) > k {
		*buf = (*buf)[:startLen+k]
	}

	return nil
}

// beamSearch performs beam search through the graph.
func (s *Segment) beamSearch(query []float32, distTable []float32, queryBQ []uint64, topK int, filter func(uint32) bool, scratch *searchScratch) []distNode {
	if len(s.graph) == 0 {
		return nil
	}

	candidates := &scratch.candidates
	*candidates = (*candidates)[:0]

	entryPoint := s.entryPoint
	if entryPoint >= uint32(len(s.graph)) {
		return nil
	}

	entryDist := s.computeDistance(query, entryPoint, distTable)
	candidates.push(distNode{id: entryPoint, dist: entryDist})

	maxID := uint32(len(s.graph))
	visited := scratch.visited
	visited.ClearAll()
	if visited.Len() < maxID {
		visited = bitset.New(maxID)
		scratch.visited = visited
	}
	visited.Set(entryPoint)

	results := scratch.beamBuf[:0]

	// Add entry point if it passes filter
	if filter == nil || filter(entryPoint) {
		results = append(results, distNode{id: entryPoint, dist: entryDist})
	}

	for len(*candidates) > 0 {
		curr := candidates.pop()

		if len(results) >= topK {
			worstInBeam := results[len(results)-1].dist
			if curr.dist > worstInBeam {
				break
			}
		}

		if curr.id < uint32(len(s.graph)) {
			for _, neighbor := range s.graph[curr.id] {
				if neighbor >= maxID || visited.Test(neighbor) {
					continue
				}
				visited.Set(neighbor)

				if filter != nil && !filter(neighbor) {
					continue
				}

				// BQ Prefilter
				if queryBQ != nil {
					words := (s.dim + 63) / 64
					offset := int(neighbor) * words
					if offset+words <= len(s.bqCodes) {
						norm := quantization.NormalizedHammingDistance(queryBQ, s.bqCodes[offset:offset+words], s.dim)
						if norm > s.opts.BinaryPrefilterMaxNormalizedDistance {
							continue
						}
					}
				}

				dist := s.computeDistance(query, neighbor, distTable)
				candidates.push(distNode{id: neighbor, dist: dist})

				results = append(results, distNode{id: neighbor, dist: dist})
				sortDistNodes(results)
				if len(results) > topK*2 {
					results = results[:topK*2]
				}
			}
		}
	}

	sortDistNodes(results)
	if len(results) > topK {
		results = results[:topK]
	}
	return results
}

func (s *Segment) rerank(query []float32, candidates []distNode, k int) []index.SearchResult {
	results := make([]distNode, 0, len(candidates))
	for _, c := range candidates {
		vec := s.getVector(c.id)
		if vec == nil {
			continue
		}
		dist := s.distFunc(query, vec)
		results = append(results, distNode{id: c.id, dist: dist})
	}

	sortDistNodes(results)
	if len(results) > k {
		results = results[:k]
	}

	out := make([]index.SearchResult, len(results))
	for i, r := range results {
		out[i] = index.SearchResult{ID: uint32(s.baseID) + r.id, Distance: r.dist}
	}
	return out
}

func (s *Segment) computeDistance(v []float32, id uint32, distTable []float32) float32 {
	M := s.opts.PQSubvectors
	offset := int(id) * M
	if distTable != nil && offset+M <= len(s.pqCodes) {
		return s.pq.AdcDistance(distTable, s.pqCodes[offset:offset+M])
	}
	vec := s.getVector(id)
	if vec == nil {
		return math.MaxFloat32
	}
	return s.distFunc(v, vec)
}

func (s *Segment) getVector(id uint32) []float32 {
	if s.mmapReader == nil {
		return nil
	}
	offset := int(id) * int(s.dim) * 4
	if offset < 0 || offset+int(s.dim)*4 > len(s.mmapReader.Data) {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&s.mmapReader.Data[offset])), s.dim)
}

// Loading methods (copied/adapted from index.go)

func (s *Segment) loadMeta(path string) error {
	f, err := os.Open(filepath.Join(path, MetaFilename))
	if err != nil {
		return err
	}
	defer f.Close()

	var header FileHeader
	if _, err := header.ReadFrom(f); err != nil {
		return err
	}
	if err := header.Validate(); err != nil {
		return err
	}

	s.fileFlags = header.Flags
	s.dim = int(header.Dimension)
	s.count = uint32(header.Count)
	s.distType = header.DistType()
	s.distFunc = index.NewDistanceFunc(s.distType)

	// Update options from header to ensure correct PQ/BQ handling
	s.opts.PQSubvectors = int(header.PQSubvectors)
	s.opts.PQCentroids = int(header.PQCentroids)

	var entryBuf [4]byte
	if _, err := io.ReadFull(f, entryBuf[:]); err != nil {
		return err
	}
	s.entryPoint = binary.LittleEndian.Uint32(entryBuf[:])

	return s.loadPQCodebooks(f, &header)
}

func (s *Segment) loadPQCodebooks(r io.Reader, header *FileHeader) error {
	M := int(header.PQSubvectors)
	K := int(header.PQCentroids)
	subDim := s.dim / M

	var err error
	s.pq, err = quantization.NewProductQuantizer(s.dim, M, K)
	if err != nil {
		return err
	}

	// Read scales
	scales := make([]float32, M)
	if err := binary.Read(r, binary.LittleEndian, scales); err != nil {
		return fmt.Errorf("read scales: %w", err)
	}

	// Read offsets
	offsets := make([]float32, M)
	if err := binary.Read(r, binary.LittleEndian, offsets); err != nil {
		return fmt.Errorf("read offsets: %w", err)
	}

	// Read codebooks (int8)
	totalInt8s := M * K * subDim
	codebooks := make([]int8, totalInt8s)
	if err := binary.Read(r, binary.LittleEndian, codebooks); err != nil {
		return fmt.Errorf("read codebooks: %w", err)
	}

	s.pq.SetCodebooks(codebooks, scales, offsets)
	return nil
}

func (s *Segment) loadGraph(path string) error {
	f, err := os.Open(filepath.Join(path, GraphFilename))
	if err != nil {
		return err
	}
	defer f.Close()

	s.graph = make([][]uint32, s.count)
	buf := make([]byte, 4)
	degBuf := make([]byte, 4)

	for i := 0; i < int(s.count); i++ {
		if _, err := io.ReadFull(f, degBuf); err != nil {
			return err
		}
		degree := binary.LittleEndian.Uint32(degBuf)
		s.graph[i] = make([]uint32, degree)
		for j := uint32(0); j < degree; j++ {
			if _, err := io.ReadFull(f, buf); err != nil {
				return err
			}
			s.graph[i][j] = binary.LittleEndian.Uint32(buf)
		}
	}
	return nil
}

func (s *Segment) loadPQCodes(path string) error {
	f, err := os.Open(filepath.Join(path, PQCodesFilename))
	if err != nil {
		return err
	}
	defer f.Close()

	M := s.opts.PQSubvectors
	s.pqCodes = make([]byte, int(s.count)*M)
	if _, err := io.ReadFull(f, s.pqCodes); err != nil {
		return err
	}
	return nil
}

func (s *Segment) loadBQCodes(path string) error {
	f, err := os.Open(filepath.Join(path, BQCodesFilename))
	if err != nil {
		return err
	}
	defer f.Close()

	words := (s.dim + 63) / 64
	s.bqCodes = make([]uint64, int(s.count)*words)
	buf := make([]byte, 8)
	for i := 0; i < len(s.bqCodes); i++ {
		if _, err := io.ReadFull(f, buf); err != nil {
			return err
		}
		s.bqCodes[i] = binary.LittleEndian.Uint64(buf)
	}
	return nil
}

// VectorByID retrieves a vector by its global ID.
func (s *Segment) VectorByID(ctx context.Context, id core.LocalID) ([]float32, error) {
	if id < s.baseID || id >= s.baseID+core.LocalID(s.count) {
		return nil, &index.ErrNodeNotFound{ID: id}
	}
	localID := uint32(id - s.baseID)
	vec := s.getVector(localID)
	if vec == nil {
		return nil, &index.ErrNodeNotFound{ID: id}
	}
	return vec, nil
}

// candidateHeap is a min-heap of distNodes, specialized to avoid interface boxing.
type candidateHeap []distNode

func (h *candidateHeap) push(n distNode) {
	*h = append(*h, n)
	h.up(len(*h) - 1)
}

func (h *candidateHeap) pop() distNode {
	old := *h
	n := len(old) - 1
	root := old[0]
	old[0] = old[n]
	*h = old[:n]
	h.down(0, len(*h))
	return root
}

func (h candidateHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || h[j].dist >= h[i].dist {
			break
		}
		h[i], h[j] = h[j], h[i]
		j = i
	}
}

func (h candidateHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && h[j2].dist < h[j1].dist {
			j = j2 // = 2*i + 2  // right child
		}
		if h[j].dist >= h[i].dist {
			break
		}
		h[i], h[j] = h[j], h[i]
		i = j
	}
}

// SearchWithContext performs KNN search using the provided Searcher context.
func (s *Segment) SearchWithContext(ctx context.Context, query []float32, k int, distTable []float32, filter func(core.LocalID) bool, sr *searcher.Searcher) error {
	// Local filter wrapper
	var localFilter func(uint32) bool
	if filter != nil {
		localFilter = func(localID uint32) bool {
			return filter(s.baseID + core.LocalID(localID))
		}
	}

	// Phase 1: Beam Search
	rerankK := s.opts.RerankK
	if rerankK < k {
		rerankK = k * 2
	}

	if distTable == nil && s.pq != nil {
		distTable = s.pq.BuildDistanceTable(query)
	}

	// Optional BQ query encoding
	var queryBQ []uint64
	if s.opts.EnableBinaryPrefilter && s.bq != nil && len(s.bqCodes) > 0 {
		queryBQ = s.bq.EncodeUint64(query)
	}

	// Use sr.ScratchCandidates for beam search exploration
	sr.ScratchCandidates.Reset()

	// Perform beam search
	// We use a local slice for results to avoid polluting sr.Candidates with PQ distances
	candidates := s.beamSearchWithContext(query, distTable, queryBQ, rerankK, localFilter, sr)

	// Phase 2: Rerank and push to sr.Candidates (MaxHeap)
	for _, c := range candidates {
		// Rerank
		vec := s.getVector(uint32(c.Node))
		if vec == nil {
			continue
		}
		dist := s.distFunc(query, vec)

		// Push to result heap (bounded by k)
		// Note: c.Node is local ID, we need global ID
		sr.Candidates.PushItemBounded(searcher.PriorityQueueItem{Node: core.LocalID(s.baseID + core.LocalID(c.Node)), Distance: dist}, k)
	}

	return nil
}

func (s *Segment) beamSearchWithContext(query []float32, distTable []float32, queryBQ []uint64, topK int, filter func(uint32) bool, sr *searcher.Searcher) []searcher.PriorityQueueItem {
	if len(s.graph) == 0 {
		return nil
	}

	entryPoint := s.entryPoint
	if entryPoint >= uint32(len(s.graph)) {
		return nil
	}

	globalEntryPoint := s.baseID + core.LocalID(entryPoint)
	// Note: We don't check if already visited because we want to start search here regardless.
	// But we mark it visited.
	sr.Visited.Visit(globalEntryPoint)

	entryDist := s.computeDistance(query, entryPoint, distTable)

	sr.ScratchCandidates.Reset()
	sr.ScratchCandidates.PushItem(searcher.PriorityQueueItem{Node: core.LocalID(entryPoint), Distance: entryDist})

	// Keep track of best results found
	results := sr.ScratchResults
	results = append(results, searcher.PriorityQueueItem{Node: core.LocalID(entryPoint), Distance: entryDist})

	for sr.ScratchCandidates.Len() > 0 {
		curr, _ := sr.ScratchCandidates.PopItem()

		// Expand neighbors
		neighbors := s.graph[curr.Node]
		for _, neighborID := range neighbors {
			globalNeighborID := s.baseID + core.LocalID(neighborID)
			if sr.Visited.Visited(globalNeighborID) {
				continue
			}
			sr.Visited.Visit(globalNeighborID)

			if filter != nil && !filter(neighborID) {
				continue
			}

			// BQ Prefilter
			if queryBQ != nil {
				words := (s.dim + 63) / 64
				offset := int(neighborID) * words
				if offset+words <= len(s.bqCodes) {
					norm := quantization.NormalizedHammingDistance(queryBQ, s.bqCodes[offset:offset+words], s.dim)
					if norm > s.opts.BinaryPrefilterMaxNormalizedDistance {
						continue
					}
				}
			}

			dist := s.computeDistance(query, neighborID, distTable)

			item := searcher.PriorityQueueItem{Node: core.LocalID(neighborID), Distance: dist}
			sr.ScratchCandidates.PushItem(item)
			results = append(results, item)
		}

		// Sort and prune results to keep size manageable
		if len(results) > topK*4 {
			sort.Slice(results, func(i, j int) bool {
				return results[i].Distance < results[j].Distance
			})
			results = results[:topK*2]
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	if len(results) > topK {
		results = results[:topK]
	}

	// Update scratch buffer to retain capacity
	sr.ScratchResults = results

	return results
}
