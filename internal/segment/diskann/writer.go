package diskann

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"math/rand"

	"github.com/hupe1980/vecgo/distance"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/resource"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
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
	r               int             // Max degree
	l               int             // Search list size
	alpha           float32         // Pruning factor
	pqSubvectors    int             // PQ M
	pqCentroids     int             // PQ K
	compressionType CompressionType // LZ4 or None

	// Data
	vectors  [][]float32
	ids      []model.ID
	metadata [][]byte
	payloads [][]byte

	// Build state
	graph            [][]uint32
	entryPoint       uint32
	quantizationType quantization.Type
	pq               *quantization.ProductQuantizer
	iq               *quantization.Int4Quantizer

	// BFS reordering permutation: addOrderRowID[i] = finalRowID
	// After Write(), this maps the original insertion order to final row positions.
	addOrderToFinalRow []uint32
	compressedVectors  [][]byte // Stores PQ codes or RaBitQ codes or INT4
	bqCodes            [][]uint64
	distFunc           distance.Func
	index              *imetadata.UnifiedIndex
}

// Options for the DiskANN writer.
type Options struct {
	R                  int
	L                  int
	Alpha              float32
	QuantizationType   quantization.Type
	PQSubvectors       int
	PQCentroids        int
	ResourceController *resource.Controller
	CompressionType    CompressionType // LZ4 or None
}

func DefaultOptions() Options {
	return Options{
		R:                64,
		L:                100,
		Alpha:            1.2,
		QuantizationType: quantization.TypeNone,
		PQSubvectors:     0, // Auto-detect or disable
		PQCentroids:      256,
		CompressionType:  CompressionLZ4, // LZ4 by default for best-in-class storage
	}
}

// NewWriter creates a new DiskANN segment writer.
func NewWriter(w io.Writer, payloadW io.Writer, segID uint64, dim int, metric distance.Metric, opts Options) *Writer {
	defaults := DefaultOptions()
	if opts.R == 0 {
		opts.R = defaults.R
	}
	if opts.L == 0 {
		opts.L = defaults.L
	}
	if opts.Alpha == 0 {
		opts.Alpha = defaults.Alpha
	}
	if opts.PQCentroids == 0 {
		opts.PQCentroids = defaults.PQCentroids
	}

	return &Writer{
		w:                w,
		payloadW:         payloadW,
		segmentID:        segID,
		dim:              dim,
		metric:           metric,
		rc:               opts.ResourceController,
		r:                opts.R,
		l:                opts.L,
		alpha:            opts.Alpha,
		quantizationType: opts.QuantizationType,
		pqSubvectors:     opts.PQSubvectors,
		pqCentroids:      opts.PQCentroids, compressionType: opts.CompressionType, vectors: make([][]float32, 0),
		ids:   make([]model.ID, 0),
		index: imetadata.NewUnifiedIndex(),
	}
}

// Add adds a vector and its ID to the segment.
func (w *Writer) Add(id model.ID, vec []float32, md metadata.Document, payload []byte) error {
	if len(vec) != w.dim {
		return errors.New("dimension mismatch")
	}
	// Copy vector to ensure ownership
	v := make([]float32, len(vec))
	copy(v, vec)
	w.vectors = append(w.vectors, v)
	w.ids = append(w.ids, id)
	w.payloads = append(w.payloads, payload)

	// Serialize metadata
	if md != nil {
		b, err := md.MarshalBinary()
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

// GetIDMapping returns a map from ID to the final RowID in the segment.
// This must be called after Write() completes to get the correct mapping
// since BFS reordering changes vector positions.
func (w *Writer) GetIDMapping() map[model.ID]uint32 {
	result := make(map[model.ID]uint32, len(w.ids))

	// If no reordering happened, addOrderToFinalRow is nil
	// In that case, the order is preserved (index i -> row i)
	if w.addOrderToFinalRow == nil {
		for i, id := range w.ids {
			result[id] = uint32(i)
		}
		return result
	}

	// After reordering, w.ids is already reordered (w.ids[newRow] = id).
	// But we need to map original Add order to final row.
	// addOrderToFinalRow[addOrderIndex] = finalRowID
	// We also need the ID that was added at each addOrderIndex.
	// But w.ids has been reordered, so w.ids[finalRowID] = id.
	// So: result[w.ids[addOrderToFinalRow[i]]] = addOrderToFinalRow[i] for all i
	// But that's circular. Let's just iterate the final state:
	// After reordering, w.ids[row] = ID at that row.
	for row, id := range w.ids {
		result[id] = uint32(row)
	}
	return result
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

	// 1. Train Quantizer
	if w.quantizationType == quantization.TypePQ && w.pqSubvectors > 0 {
		if err := w.trainPQ(ctx); err != nil {
			return fmt.Errorf("train PQ: %w", err)
		}
	} else if w.quantizationType == quantization.TypeRaBitQ {
		if err := w.trainRaBitQ(ctx); err != nil {
			return fmt.Errorf("train RaBitQ: %w", err)
		}
	} else if w.quantizationType == quantization.TypeINT4 {
		if err := w.trainInt4(ctx); err != nil {
			return fmt.Errorf("train INT4: %w", err)
		}
	}

	// 2. Build Graph (Vamana)
	if err := w.buildGraph(ctx); err != nil {
		return fmt.Errorf("build graph: %w", err)
	}

	// 2.5. Optimize Graph Layout (BFS Reordering)
	// This improves search locality by 20-40%.
	if err := w.reorderBFS(ctx); err != nil {
		return fmt.Errorf("reorder graph: %w", err)
	}

	// 3. Write to disk
	err := w.Flush()

	// Release memory if controller is present
	if w.rc != nil {
		// Release graph memory
		graphSize := int64(len(w.vectors) * w.r * 4)
		w.rc.ReleaseMemory(graphSize)

		// Release Quantization memory
		if len(w.compressedVectors) > 0 {
			quantSize := int64(len(w.vectors) * len(w.compressedVectors[0]))
			w.rc.ReleaseMemory(quantSize)
		}
	}

	return err
}

func (w *Writer) trainPQ(ctx context.Context) error {
	// Simple heuristic: if not enough vectors, don't use PQ or reduce params
	if len(w.vectors) < w.pqCentroids {
		w.quantizationType = quantization.TypeNone // Disable PQ
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

	w.compressedVectors = make([][]byte, len(w.vectors))
	for i, vec := range w.vectors {
		w.compressedVectors[i], err = pq.Encode(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

func (w *Writer) trainRaBitQ(ctx context.Context) error {
	rq := quantization.NewRaBitQuantizer(w.dim)

	// Estimate memory: (Dim/64)*8 + 4 bytes per vector
	bytesPerVec := ((w.dim+63)/64)*8 + 4
	if w.rc != nil {
		size := int64(len(w.vectors) * bytesPerVec)
		if err := w.rc.AcquireMemory(ctx, size); err != nil {
			return err
		}
	}

	w.compressedVectors = make([][]byte, len(w.vectors))
	var err error
	for i, vec := range w.vectors {
		w.compressedVectors[i], err = rq.Encode(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

func (w *Writer) trainInt4(ctx context.Context) error {
	iq := quantization.NewInt4Quantizer(w.dim)
	if err := iq.Train(w.vectors); err != nil {
		return err
	}
	w.iq = iq

	// Estimate memory: ceil(dim/2) bytes per vector
	bytesPerVec := (w.dim + 1) / 2
	if w.rc != nil {
		size := int64(len(w.vectors) * bytesPerVec)
		if err := w.rc.AcquireMemory(ctx, size); err != nil {
			return err
		}
	}

	w.compressedVectors = make([][]byte, len(w.vectors))
	var err error
	for i, vec := range w.vectors {
		w.compressedVectors[i], err = iq.Encode(vec)
		if err != nil {
			return err
		}
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
		// Linear Scan for best unexpanded
		// Optimization: Don't sort everything every time. Just find best unexpanded.
		// However, we need to know if it's within Top-L.
		// Sorting is easiest but O(N log N).

		sortDistNodes(pool)

		// Find closest unexpanded
		var currIdx int = -1
		for i := range pool {
			if !expanded[pool[i].id] {
				currIdx = i
				break
			}
		}

		if currIdx == -1 {
			break // All top nodes expanded
		}

		if currIdx >= l {
			// Best unexpanded is outside top L?
			// Technically Vamana says explore if within L.
			// But if we have > L nodes, pool[L] is the L-th best.
			// If pool[currIdx].dist > pool[l-1].dist, we stop.
			if len(pool) > l {
				break
			}
		}

		current := pool[currIdx]
		expanded[current.id] = true

		if len(pool) > l+50 { // Prune if growing too large
			pool = pool[:l+50]
		}

		// Add neighbors
		neighbors := w.graph[current.id]
		for _, neighbor := range neighbors {
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

	// Compressed Vectors (PQ or RaBitQ)
	compressedVectorsSize := uint64(0)
	if len(w.compressedVectors) > 0 {
		compressedVectorsSize = uint64(len(w.vectors) * len(w.compressedVectors[0]))
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
		QuantizationType: uint8(w.quantizationType),
		CompressionType:  uint8(w.compressionType),
	}
	if w.quantizationType == quantization.TypePQ {
		h.PQSubvectors = uint16(w.pqSubvectors)
		h.PQCentroids = uint16(w.pqCentroids)
	}

	// Write header (will overwrite later with correct offsets)
	if _, err := bw.Write(h.Encode()); err != nil {
		return err
	}

	// Suppress unused variable warnings - these are used for offset calculation
	_ = vectorSize
	_ = graphSize
	_ = compressedVectorsSize

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

	// Write Compressed Vectors (PQ or RaBitQ or INT4)
	if len(w.compressedVectors) > 0 {
		if w.quantizationType == quantization.TypePQ || w.quantizationType == quantization.TypeINT4 {
			h.PQCodesOffset = bytesWritten
		} else if w.quantizationType == quantization.TypeRaBitQ {
			h.BQCodesOffset = bytesWritten
		}

		for _, code := range w.compressedVectors {
			if _, err := mw.Write(code); err != nil {
				return err
			}
		}
		bytesWritten += compressedVectorsSize

		// Write PQ Codebooks if PQ
		if w.quantizationType == quantization.TypePQ && w.pq != nil {
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
		} else if w.quantizationType == quantization.TypeINT4 && w.iq != nil {
			// Write INT4 params (Min, Diff)
			h.PQCodebookOffset = bytesWritten
			b, err := w.iq.MarshalBinary()
			if err != nil {
				return err
			}
			if _, err := mw.Write(b); err != nil {
				return err
			}
			bytesWritten += uint64(len(b))
		}
	}

	// Write IDs
	h.PKOffset = bytesWritten

	// Calculate blob
	var pkBlob bytes.Buffer
	pkBlob.Grow(len(w.ids) * 8)

	for _, id := range w.ids {
		binary.Write(&pkBlob, binary.LittleEndian, uint64(id))
	}

	// Write Blob
	if _, err := mw.Write(pkBlob.Bytes()); err != nil {
		return err
	}
	bytesWritten += uint64(pkBlob.Len())

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
