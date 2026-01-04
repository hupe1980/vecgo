package diskann

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/rand"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/resource"
)

// Writer builds a DiskANN segment.
type Writer struct {
	w         io.Writer
	payloadW  io.Writer
	segmentID uint64
	dim       int
	metric    distance.Metric
	rc        *resource.Controller

	// Configuration
	r            int     // Max degree
	l            int     // Search list size
	alpha        float32 // Pruning factor
	pqSubvectors int     // PQ M
	pqCentroids  int     // PQ K

	// Data
	vectors  [][]float32
	pks      []uint64
	metadata [][]byte
	payloads [][]byte

	// Build state
	graph      [][]uint32
	entryPoint uint32
	pq         *quantization.ProductQuantizer
	pqCodes    [][]byte
	bqCodes    [][]uint64
	distFunc   distance.Func
	index      *metadata.UnifiedIndex
}

// Options for the DiskANN writer.
type Options struct {
	R                  int
	L                  int
	Alpha              float32
	PQSubvectors       int
	PQCentroids        int
	ResourceController *resource.Controller
}

func DefaultOptions() Options {
	return Options{
		R:            64,
		L:            100,
		Alpha:        1.2,
		PQSubvectors: 0, // Auto-detect or disable
		PQCentroids:  256,
	}
}

// NewWriter creates a new DiskANN segment writer.
func NewWriter(w io.Writer, payloadW io.Writer, segID uint64, dim int, metric distance.Metric, opts Options) *Writer {
	if opts.R == 0 {
		opts = DefaultOptions()
	}
	return &Writer{
		w:            w,
		payloadW:     payloadW,
		segmentID:    segID,
		dim:          dim,
		metric:       metric,
		rc:           opts.ResourceController,
		r:            opts.R,
		l:            opts.L,
		alpha:        opts.Alpha,
		pqSubvectors: opts.PQSubvectors,
		pqCentroids:  opts.PQCentroids,
		vectors:      make([][]float32, 0),
		pks:          make([]uint64, 0),
		index:        metadata.NewUnifiedIndex(),
	}
}

// Add adds a vector and its PK to the segment.
func (w *Writer) Add(pk uint64, vec []float32, md metadata.Document, payload []byte) error {
	if len(vec) != w.dim {
		return errors.New("dimension mismatch")
	}
	// Copy vector to ensure ownership
	v := make([]float32, len(vec))
	copy(v, vec)
	w.vectors = append(w.vectors, v)
	w.pks = append(w.pks, pk)
	w.payloads = append(w.payloads, payload)

	// Serialize metadata
	if md != nil {
		b, err := json.Marshal(md.ToMap())
		if err != nil {
			return err
		}
		w.metadata = append(w.metadata, b)
		w.index.AddInvertedIndex(model.RowID(len(w.vectors)-1), md)
	} else {
		w.metadata = append(w.metadata, nil)
	}
	return nil
}

// Write builds the graph and writes the segment to the underlying writer.
func (w *Writer) Write(ctx context.Context) error {
	if len(w.vectors) == 0 {
		return errors.New("no vectors to write")
	}

	if w.rc != nil {
		if err := w.rc.AcquireBackground(ctx); err != nil {
			return err
		}
		defer w.rc.ReleaseBackground()
	}

	// 1. Train PQ if enabled
	if w.pqSubvectors > 0 {
		if err := w.trainPQ(ctx); err != nil {
			return fmt.Errorf("train PQ: %w", err)
		}
	}

	// 2. Build Graph (Vamana)
	if err := w.buildGraph(ctx); err != nil {
		return fmt.Errorf("build graph: %w", err)
	}

	// 3. Write to disk
	err := w.Flush()

	// Release memory if controller is present
	if w.rc != nil {
		// Release graph memory
		graphSize := int64(len(w.vectors) * w.r * 4)
		w.rc.ReleaseMemory(graphSize)

		// Release PQ memory if used
		if w.pqSubvectors > 0 {
			pqSize := int64(len(w.vectors) * w.pqSubvectors)
			w.rc.ReleaseMemory(pqSize)
		}
	}

	return err
}

func (w *Writer) trainPQ(ctx context.Context) error {
	// Simple heuristic: if not enough vectors, don't use PQ or reduce params
	if len(w.vectors) < w.pqCentroids {
		w.pqSubvectors = 0 // Disable PQ
		return nil
	}

	pq, err := quantization.NewProductQuantizer(
		w.dim,
		w.pqSubvectors,
		w.pqCentroids,
	)
	if err != nil {
		return err
	}

	if err := pq.Train(w.vectors); err != nil {
		return err
	}
	w.pq = pq

	if w.rc != nil {
		size := int64(len(w.vectors) * w.pqSubvectors)
		if err := w.rc.AcquireMemory(ctx, size); err != nil {
			return err
		}
	}

	w.pqCodes = make([][]byte, len(w.vectors))
	for i, vec := range w.vectors {
		w.pqCodes[i] = pq.Encode(vec)
	}
	return nil
}

func (w *Writer) buildGraph(ctx context.Context) error {
	distFunc, err := distance.Provider(w.metric)
	if err != nil {
		return err
	}
	w.distFunc = distFunc

	n := len(w.vectors)

	if w.rc != nil {
		// Estimate memory: n * r * 4 bytes
		estimatedBytes := int64(n * w.r * 4)
		if err := w.rc.AcquireMemory(ctx, estimatedBytes); err != nil {
			return err
		}
	}

	w.graph = make([][]uint32, n)

	// Initialize random graph
	for i := 0; i < n; i++ {
		w.graph[i] = make([]uint32, 0, w.r)
	}

	// Calculate centroid to find entry point
	centroid := make([]float32, w.dim)
	for _, vec := range w.vectors {
		for j, val := range vec {
			centroid[j] += val
		}
	}
	for j := range centroid {
		centroid[j] /= float32(n)
	}

	// Find entry point (closest to centroid)
	minDist := float32(math.MaxFloat32)
	for i, vec := range w.vectors {
		d := w.dist(vec, centroid)
		if d < minDist {
			minDist = d
			w.entryPoint = uint32(i)
		}
	}

	// Vamana construction
	// Pass 1: Random initialization (implicit in empty graph? No, usually random edges)
	// Actually, Vamana starts with random graph or empty.
	// Standard Vamana:
	// 1. Initialize random graph (R edges per node)
	// 2. For each node, search for closest candidates starting from entry point
	// 3. Prune edges

	// Initialize with random neighbors
	for i := 0; i < n; i++ {
		perm := rand.Perm(n)
		count := 0
		for _, idx := range perm {
			if uint32(idx) == uint32(i) {
				continue
			}
			w.graph[i] = append(w.graph[i], uint32(idx))
			count++
			if count >= w.r {
				break
			}
		}
	}

	// Two passes usually
	for pass := 0; pass < 2; pass++ {
		alpha := w.alpha
		if pass == 0 {
			alpha = 1.0 // First pass with alpha=1
		}

		for i := 0; i < n; i++ {
			// Greedy search to find candidates
			candidates := w.greedySearch(w.vectors[i], w.entryPoint, w.l)

			// Robust prune
			w.graph[i] = w.robustPrune(uint32(i), candidates, w.r, alpha)

			// Add back-edges
			for _, neighbor := range w.graph[i] {
				w.addBackEdge(neighbor, uint32(i), w.r, alpha)
			}
		}
	}

	return nil
}

func (w *Writer) dist(v1, v2 []float32) float32 {
	if w.distFunc == nil {
		// Fallback or panic. Should be set in buildGraph.
		// For safety, set it if nil (though buildGraph sets it)
		f, _ := distance.Provider(w.metric)
		w.distFunc = f
	}
	return w.distFunc(v1, v2)
}

func (w *Writer) greedySearch(query []float32, startNode uint32, l int) []uint32 {
	// Simple greedy search implementation
	visited := make(map[uint32]bool)
	visited[startNode] = true

	candidates := make([]distNode, 0, l*2)
	candidates = append(candidates, distNode{id: startNode, dist: w.dist(w.vectors[startNode], query)})

	// Use a heap for best candidates
	// For construction, we just need a set of candidates.
	// Standard greedy search:
	// Maintain a set of L closest nodes found so far.
	// Expand the closest unexpanded node.

	// Simplified for brevity (and correctness):
	// 1. Start with entry point
	// 2. While we have unexpanded nodes in the top L:
	//    a. Pick closest unexpanded
	//    b. Add its neighbors
	//    c. Update top L

	// Using a simple slice and sorting for now (can be optimized with heaps)

	// Initial set
	pool := make([]distNode, 0, l*2)
	pool = append(pool, distNode{id: startNode, dist: w.dist(w.vectors[startNode], query)})

	expanded := make(map[uint32]bool)

	for {
		// Sort pool
		sortDistNodes(pool)

		// Find closest unexpanded
		var current *distNode
		for i := range pool {
			if !expanded[pool[i].id] {
				current = &pool[i]
				break
			}
		}

		if current == nil {
			break // All top nodes expanded
		}

		// If current is further than the L-th node, we can stop?
		// Vamana paper says: iterate until all candidates in L are expanded.
		if len(pool) > l && current.dist > pool[l-1].dist {
			// Optimization: if the closest unexpanded is worse than the L-th best,
			// and we have at least L nodes, we can stop.
			break
		}

		expanded[current.id] = true

		// Add neighbors
		for _, neighbor := range w.graph[current.id] {
			if !visited[neighbor] {
				visited[neighbor] = true
				d := w.dist(w.vectors[neighbor], query)
				pool = append(pool, distNode{id: neighbor, dist: d})
			}
		}
	}

	sortDistNodes(pool)
	if len(pool) > l {
		pool = pool[:l]
	}

	res := make([]uint32, len(pool))
	for i, p := range pool {
		res[i] = p.id
	}
	return res
}

func (w *Writer) robustPrune(node uint32, candidates []uint32, r int, alpha float32) []uint32 {
	// Add current neighbors to candidates
	// (In Vamana, we prune the union of current neighbors and search results)
	// The candidates passed here should already include them or we add them.
	// In buildGraph, I passed search results. I should merge with existing neighbors.

	unique := make(map[uint32]bool)
	for _, c := range candidates {
		unique[c] = true
	}
	for _, n := range w.graph[node] {
		unique[n] = true
	}
	delete(unique, node) // Remove self

	cands := make([]distNode, 0, len(unique))
	nodeVec := w.vectors[node]
	for id := range unique {
		cands = append(cands, distNode{id: id, dist: w.dist(w.vectors[id], nodeVec)})
	}

	sortDistNodes(cands)

	selected := make([]uint32, 0, r)
	for _, c := range cands {
		if len(selected) >= r {
			break
		}

		// Check diversity
		diverse := true
		for _, s := range selected {
			distCS := w.dist(w.vectors[c.id], w.vectors[s])
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

func (w *Writer) addBackEdge(src, dst uint32, r int, alpha float32) {
	// Check if exists
	for _, n := range w.graph[src] {
		if n == dst {
			return
		}
	}

	w.graph[src] = append(w.graph[src], dst)
	if len(w.graph[src]) > r {
		// Prune
		// Note: robustPrune needs candidates. Here we just have the current graph.
		// We treat the current graph as the candidate set.
		candidates := make([]uint32, len(w.graph[src]))
		copy(candidates, w.graph[src])
		w.graph[src] = w.robustPrune(src, candidates, r, alpha)
	}
}

func (w *Writer) Flush() error {
	// Calculate offsets

	// Vectors
	vectorSize := uint64(len(w.vectors) * w.dim * 4)

	// Graph
	// Fixed size: N * R * 4 bytes
	graphSize := uint64(len(w.vectors) * w.r * 4)

	// PQ Codes
	pqCodesSize := uint64(0)
	if w.pqSubvectors > 0 {
		pqCodesSize = uint64(len(w.vectors) * w.pqSubvectors)
	}

	// We'll use a buffered writer
	bw := bufio.NewWriterSize(w.w, 4*1024*1024) // 4MB buffer

	// Placeholder header
	h := FileHeader{
		Magic:            MagicNumber,
		Version:          Version,
		SegmentID:        w.segmentID,
		RowCount:         uint32(len(w.vectors)),
		Dim:              uint32(w.dim),
		Metric:           uint8(w.metric),
		MaxDegree:        uint32(w.r),
		SearchListSize:   uint32(w.l),
		Entrypoint:       w.entryPoint,
		QuantizationType: 0,
	}
	if w.pqSubvectors > 0 {
		h.QuantizationType = 2 // PQ
		h.PQSubvectors = uint16(w.pqSubvectors)
		h.PQCentroids = uint16(w.pqCentroids)
	}

	// Write header (will overwrite later with correct offsets)
	if _, err := bw.Write(h.Encode()); err != nil {
		return err
	}

	// Calculate Checksum of the body
	crc := crc32.New(crc32.MakeTable(crc32.Castagnoli))
	mw := io.MultiWriter(bw, crc)

	// Write Vectors
	h.VectorOffset = uint64(HeaderSize) // Assuming we are at start
	// Actually, we should track bytes written.
	bytesWritten := uint64(HeaderSize)

	for _, vec := range w.vectors {
		for _, v := range vec {
			if err := binary.Write(mw, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}
	bytesWritten += vectorSize

	// Write Graph
	h.GraphOffset = bytesWritten
	for i := 0; i < len(w.vectors); i++ {
		var neighbors []uint32
		if i < len(w.graph) {
			neighbors = w.graph[i]
		}
		// Write neighbors, padded to R
		for j := 0; j < w.r; j++ {
			val := uint32(0xFFFFFFFF) // Sentinel
			if j < len(neighbors) {
				val = neighbors[j]
			}
			if err := binary.Write(mw, binary.LittleEndian, val); err != nil {
				return err
			}
		}
	}
	bytesWritten += graphSize

	// Write PQ Codes
	if w.pqSubvectors > 0 {
		h.PQCodesOffset = bytesWritten
		for _, code := range w.pqCodes {
			if _, err := mw.Write(code); err != nil {
				return err
			}
		}
		bytesWritten += pqCodesSize

		// Write PQ Codebooks
		h.PQCodebookOffset = bytesWritten
		// Serialize PQ
		codebooks, scales, offsets := w.pq.Codebooks()

		// Write scales
		if err := binary.Write(mw, binary.LittleEndian, scales); err != nil {
			return err
		}
		bytesWritten += uint64(len(scales) * 4)

		// Write offsets
		if err := binary.Write(mw, binary.LittleEndian, offsets); err != nil {
			return err
		}
		bytesWritten += uint64(len(offsets) * 4)

		// Write codebooks
		if err := binary.Write(mw, binary.LittleEndian, codebooks); err != nil {
			return err
		}
		bytesWritten += uint64(len(codebooks)) // int8 = 1 byte
	}

	// Write PKs
	h.PKOffset = bytesWritten
	for _, pk := range w.pks {
		if err := binary.Write(mw, binary.LittleEndian, pk); err != nil {
			return err
		}
	}
	bytesWritten += uint64(len(w.pks) * 8)

	// Write Metadata
	h.MetadataOffset = bytesWritten
	// 1. Calculate offsets
	mdOffsets := make([]uint64, len(w.vectors)+1)
	currentOffset := uint64(0)
	for i, md := range w.metadata {
		mdOffsets[i] = currentOffset
		if md != nil {
			currentOffset += uint64(len(md))
		}
	}
	mdOffsets[len(w.vectors)] = currentOffset

	// 2. Write Offsets
	if err := binary.Write(mw, binary.LittleEndian, mdOffsets); err != nil {
		return err
	}
	bytesWritten += uint64(len(mdOffsets) * 8)

	// 3. Write Data
	for _, md := range w.metadata {
		if md != nil {
			if _, err := mw.Write(md); err != nil {
				return err
			}
		}
	}
	bytesWritten += currentOffset

	// Write Metadata Index
	h.MetadataIndexOffset = bytesWritten
	cw := &countingWriter{w: mw}
	if err := w.index.WriteInvertedIndex(cw); err != nil {
		return err
	}
	bytesWritten += cw.n

	// Flush
	if err := bw.Flush(); err != nil {
		return err
	}

	// If the underlying writer supports Sync, call it.
	if s, ok := w.w.(interface{ Sync() error }); ok {
		if err := s.Sync(); err != nil {
			return err
		}
	}

	// Seek back and write header
	if seeker, ok := w.w.(io.Seeker); ok {
		h.Checksum = crc.Sum32()
		if _, err := seeker.Seek(0, io.SeekStart); err != nil {
			return err
		}
		if _, err := w.w.Write(h.Encode()); err != nil {
			return err
		}
	}

	// Write payloads
	if w.payloadW != nil {
		// 1. Count
		if err := binary.Write(w.payloadW, binary.LittleEndian, uint32(len(w.payloads))); err != nil {
			return err
		}

		// 2. Offsets
		offset := uint64(0)
		for _, p := range w.payloads {
			if err := binary.Write(w.payloadW, binary.LittleEndian, offset); err != nil {
				return err
			}
			offset += uint64(len(p))
		}
		// Write total size as last offset (so we can calculate size of last element)
		if err := binary.Write(w.payloadW, binary.LittleEndian, offset); err != nil {
			return err
		}

		// 3. Data
		for _, p := range w.payloads {
			if _, err := w.payloadW.Write(p); err != nil {
				return err
			}
		}
	}

	return nil
}

type countingWriter struct {
	w io.Writer
	n uint64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += uint64(n)
	return n, err
}
