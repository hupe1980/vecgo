package diskann

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"math"
	"unsafe"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/cache"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/quantization"
	"github.com/hupe1980/vecgo/searcher"
)

// Segment implements an immutable DiskANN segment.
type Segment struct {
	header         *FileHeader
	blob           blobstore.Blob
	payloadBlob    blobstore.Blob
	data           []byte
	cache          cache.BlockCache
	verifyChecksum bool

	// Data pointers
	vectors []float32 // Full precision vectors
	pks     []uint64  // Primary keys
	graph   []uint32  // Adjacency list (flat)
	pqCodes []byte    // PQ codes

	// Metadata
	metadataOffsets []uint64
	index           *metadata.UnifiedIndex

	// Payload
	payloadCount   uint32
	payloadOffsets []uint64

	// Helpers
	pq       *quantization.ProductQuantizer
	distFunc distance.Func
}

// Option configures a Segment.
type Option func(*Segment)

// WithBlockCache sets the block cache for the segment.
func WithBlockCache(c cache.BlockCache) Option {
	return func(s *Segment) {
		s.cache = c
	}
}

// WithVerifyChecksum enables full checksum verification on open.
func WithVerifyChecksum(verify bool) Option {
	return func(s *Segment) {
		s.verifyChecksum = verify
	}
}

// WithPayloadBlob sets the payload blob for the segment.
func WithPayloadBlob(blob blobstore.Blob) Option {
	return func(s *Segment) {
		s.payloadBlob = blob
	}
}

// Open opens a DiskANN segment from a blob.
func Open(blob blobstore.Blob, opts ...Option) (*Segment, error) {
	var data []byte
	if m, ok := blob.(blobstore.Mappable); ok {
		var err error
		data, err = m.Bytes()
		if err != nil {
			return nil, err
		}
	} else {
		// Fallback: read fully into memory
		size := blob.Size()
		data = make([]byte, size)
		if _, err := blob.ReadAt(data, 0); err != nil {
			return nil, err
		}
	}

	s := &Segment{
		blob: blob,
		data: data,
	}

	for _, opt := range opts {
		opt(s)
	}

	if err := s.load(); err != nil {
		s.Close()
		return nil, err
	}

	return s, nil
}

func (s *Segment) load() error {
	if len(s.data) < HeaderSize {
		return errors.New("file too small")
	}

	var err error
	s.header, err = DecodeHeader(s.data)
	if err != nil {
		return err
	}

	if s.payloadBlob != nil {
		// Load payload offsets
		// Format: [Count uint32][Offsets uint64...]
		// Read header
		header := make([]byte, 4)
		if _, err := s.payloadBlob.ReadAt(header, 0); err != nil {
			return fmt.Errorf("failed to read payload header: %w", err)
		}
		count := binary.LittleEndian.Uint32(header)
		s.payloadCount = count

		// Read offsets
		offsetsSize := int(count+1) * 8
		offsetsBytes := make([]byte, offsetsSize)
		if _, err := s.payloadBlob.ReadAt(offsetsBytes, 4); err != nil {
			return fmt.Errorf("failed to read payload offsets: %w", err)
		}

		s.payloadOffsets = make([]uint64, count+1)
		for i := 0; i < int(count+1); i++ {
			s.payloadOffsets[i] = binary.LittleEndian.Uint64(offsetsBytes[i*8:])
		}
	}

	if s.verifyChecksum && s.header.Checksum != 0 {
		// Verify checksum (CRC32C of body)
		if len(s.data) > HeaderSize {
			body := s.data[HeaderSize:]
			sum := crc32.Checksum(body, crc32.MakeTable(crc32.Castagnoli))
			if sum != s.header.Checksum {
				return fmt.Errorf("checksum mismatch: expected %x, got %x", s.header.Checksum, sum)
			}
		}
	}

	// Validate size
	// We relax this check because we might have appended metadata or other sections
	minSize := s.header.PKOffset + uint64(s.header.RowCount)*8
	if uint64(len(s.data)) < minSize {
		return fmt.Errorf("file size too small: expected at least %d, got %d", minSize, len(s.data))
	}

	// Setup vectors pointer
	if s.header.VectorOffset+uint64(s.header.RowCount)*uint64(s.header.Dim)*4 > uint64(len(s.data)) {
		return errors.New("vector section out of bounds")
	}
	// Unsafe cast to []float32
	// Note: This assumes little-endian architecture matching file format
	vectorBytes := s.data[s.header.VectorOffset : s.header.VectorOffset+uint64(s.header.RowCount)*uint64(s.header.Dim)*4]
	header := (*unsafe.Pointer)(unsafe.Pointer(&s.vectors))
	*header = unsafe.Pointer(&vectorBytes[0])
	// We need to set length/cap manually for the slice header?
	// Go slice header: Data, Len, Cap.
	// But we can't easily modify slice header portably without reflect or unsafe.Slice (Go 1.17+).
	// Go 1.24 is used, so unsafe.Slice is available.
	s.vectors = unsafe.Slice((*float32)(unsafe.Pointer(&vectorBytes[0])), int(s.header.RowCount)*int(s.header.Dim))

	// Setup PKs
	pkSize := uint64(s.header.RowCount) * 8
	if s.header.PKOffset+pkSize > uint64(len(s.data)) {
		return errors.New("PK section out of bounds")
	}
	pkBytes := s.data[s.header.PKOffset : s.header.PKOffset+pkSize]
	s.pks = unsafe.Slice((*uint64)(unsafe.Pointer(&pkBytes[0])), int(s.header.RowCount))

	// Setup Metadata
	if s.header.MetadataOffset > 0 {
		offsetsSize := uint64(s.header.RowCount+1) * 8
		if s.header.MetadataOffset+offsetsSize > uint64(len(s.data)) {
			return errors.New("metadata offsets section out of bounds")
		}
		offsetsBytes := s.data[s.header.MetadataOffset : s.header.MetadataOffset+offsetsSize]
		s.metadataOffsets = unsafe.Slice((*uint64)(unsafe.Pointer(&offsetsBytes[0])), int(s.header.RowCount)+1)
	}

	// Setup Metadata Index
	if s.header.MetadataIndexOffset > 0 {
		if s.header.MetadataIndexOffset > uint64(len(s.data)) {
			return errors.New("metadata index offset out of bounds")
		}
		r := bytes.NewReader(s.data[s.header.MetadataIndexOffset:])
		s.index = metadata.NewUnifiedIndex()
		if err := s.index.ReadInvertedIndex(r); err != nil {
			return fmt.Errorf("failed to read metadata index: %w", err)
		}
		s.index.SetDocumentProvider(func(id model.RowID) (metadata.Document, bool) {
			doc, err := s.readMetadata(uint32(id))
			if err != nil || doc == nil {
				return nil, false
			}
			return doc, true
		})
	}

	// Setup graph pointer
	graphSize := uint64(s.header.RowCount) * uint64(s.header.MaxDegree) * 4
	if s.header.GraphOffset+graphSize > uint64(len(s.data)) {
		return errors.New("graph section out of bounds")
	}
	graphBytes := s.data[s.header.GraphOffset : s.header.GraphOffset+graphSize]
	s.graph = unsafe.Slice((*uint32)(unsafe.Pointer(&graphBytes[0])), int(s.header.RowCount)*int(s.header.MaxDegree))

	// Setup PQ if enabled
	if s.header.QuantizationType == 2 { // PQ
		if err := s.loadPQ(); err != nil {
			return err
		}
	}

	// Setup distance function
	// Metric is in header
	s.distFunc, err = distance.Provider(distance.Metric(s.header.Metric))
	if err != nil {
		return err
	}

	return nil
}

func (s *Segment) loadPQ() error {
	m := int(s.header.PQSubvectors)
	k := int(s.header.PQCentroids)
	subDim := int(s.header.Dim) / m

	// PQ Codes
	pqCodesSize := uint64(s.header.RowCount) * uint64(m)
	if s.header.PQCodesOffset+pqCodesSize > uint64(len(s.data)) {
		return errors.New("PQ codes section out of bounds")
	}
	s.pqCodes = s.data[s.header.PQCodesOffset : s.header.PQCodesOffset+pqCodesSize]

	// PQ Codebooks
	offset := s.header.PQCodebookOffset

	// Scales (M * 4 bytes)
	scalesSize := uint64(m * 4)
	if offset+scalesSize > uint64(len(s.data)) {
		return errors.New("PQ scales out of bounds")
	}
	scales := make([]float32, m)
	for i := 0; i < m; i++ {
		bits := binary.LittleEndian.Uint32(s.data[offset+uint64(i*4):])
		scales[i] = math.Float32frombits(bits)
	}
	offset += scalesSize

	// Offsets (M * 4 bytes)
	offsetsSize := uint64(m * 4)
	if offset+offsetsSize > uint64(len(s.data)) {
		return errors.New("PQ offsets out of bounds")
	}
	offsets := make([]float32, m)
	for i := 0; i < m; i++ {
		bits := binary.LittleEndian.Uint32(s.data[offset+uint64(i*4):])
		offsets[i] = math.Float32frombits(bits)
	}
	offset += offsetsSize

	// Codebooks (M * K * SubDim * 1 byte)
	codebooksSize := uint64(m * k * subDim)
	if offset+codebooksSize > uint64(len(s.data)) {
		return errors.New("PQ codebooks out of bounds")
	}
	codebooks := make([]int8, m*k*subDim)
	cbBytes := s.data[offset : offset+codebooksSize]
	for i, b := range cbBytes {
		codebooks[i] = int8(b)
	}

	pq, err := quantization.NewProductQuantizer(int(s.header.Dim), m, k)
	if err != nil {
		return err
	}
	pq.SetCodebooks(codebooks, scales, offsets)
	s.pq = pq

	return nil
}

// ID returns the segment ID.
func (s *Segment) ID() model.SegmentID {
	return model.SegmentID(s.header.SegmentID)
}

// RowCount returns the number of rows in the segment.
func (s *Segment) RowCount() uint32 {
	return s.header.RowCount
}

// Metric returns the distance metric.
func (s *Segment) Metric() distance.Metric {
	return distance.Metric(s.header.Metric)
}

// Get returns the vector for the given ID.
func (s *Segment) Get(id uint32) ([]float32, error) {
	if id >= s.header.RowCount {
		return nil, errors.New("id out of bounds")
	}
	dim := int(s.header.Dim)
	start := int(id) * dim
	return s.vectors[start : start+dim], nil
}

// Search performs an approximate nearest neighbor search.
func (s *Segment) Search(ctx context.Context, q []float32, k int, filter segment.Filter, opts model.SearchOptions, searcherCtx *searcher.Searcher) error {
	// Use RefineFactor as a proxy for search breadth.
	l := k + 100
	if opts.RefineFactor > 0 {
		l = int(float32(k) * opts.RefineFactor)
		if l < k {
			l = k
		}
	}

	return s.searchInternal(q, k, l, filter, opts.Filter, searcherCtx)
}

func (s *Segment) searchInternal(query []float32, k int, l int, filter segment.Filter, metadataFilter interface{}, searcherCtx *searcher.Searcher) error {
	if l < k {
		l = k
	}

	// Metadata filter
	var metadataFilterFn func(model.RowID) bool
	if s.index != nil && metadataFilter != nil {
		if fs, ok := metadataFilter.(*metadata.FilterSet); ok {
			metadataFilterFn = s.index.CreateStreamingFilter(fs)
		}
	}

	// Distance function wrapper
	var distFn func(id uint32) float32
	if s.pq != nil {
		m := int(s.header.PQSubvectors)
		distFn = func(id uint32) float32 {
			code := s.pqCodes[int(id)*m : int(id+1)*m]
			return s.pq.ComputeAsymmetricDistance(query, code)
		}
	} else {
		distFn = func(id uint32) float32 {
			vec, _ := s.Get(id)
			return s.distFunc(query, vec)
		}
	}

	var sc *searcher.Searcher
	if searcherCtx != nil {
		sc = searcherCtx
	} else {
		sc = searcher.Get()
		defer searcher.Put(sc)
		sc.Heap.Reset(s.Metric() != distance.MetricL2)
	}

	sc.Visited.Reset()
	sc.Candidates.Reset() // MinHeap for traversal
	// sc.Heap is already initialized by caller (engine) or above

	// Greedy search
	startNode := s.header.Entrypoint

	sc.Visited.Visit(model.RowID(startNode))
	startDist := distFn(startNode)

	sc.Candidates.PushItem(searcher.PriorityQueueItem{Node: model.RowID(startNode), Distance: startDist})

	// Helper to push to heap
	pushToHeap := func(id uint32, dist float32) {
		if filter != nil && !filter.Matches(id) {
			return
		}
		if metadataFilterFn != nil && !metadataFilterFn(model.RowID(id)) {
			return
		}

		c := model.Candidate{Loc: model.Location{SegmentID: s.ID(), RowID: model.RowID(id)}, Score: dist, Approx: true}
		if sc.Heap.Len() < k {
			sc.Heap.Push(c)
		} else {
			top := sc.Heap.Candidates[0]
			if searcher.CandidateBetter(c, top, sc.Heap.Descending()) {
				sc.Heap.ReplaceTop(c)
			}
		}
	}

	// Reset heap for local search if we were not passed a context?
	// No, if we created it locally, we reset it.
	// If passed from engine, it accumulates.

	// But wait, DiskANN uses a beam search (L) which is larger than K.
	// And it maintains a set of candidates.
	// The `sc.Heap` is the GLOBAL top-K.
	// DiskANN needs a LOCAL top-L to guide the search?
	// The original code used `candidates` slice as a pool.

	// Current implementation uses `sc.Candidates` (PriorityQueue) for traversal.
	// And `sc.Heap` for results.

	// Let's stick to the previous logic but push to `sc.Heap` instead of returning.

	// Re-add start node correctly
	// Clear heap if local? No, we want to accumulate.

	pushToHeap(startNode, startDist)

	for sc.Candidates.Len() > 0 {
		// Pop closest unexpanded
		curr, _ := sc.Candidates.PopItem()

		// Pruning
		if sc.Heap.Len() >= k { // Use K or capacity?
			// We should use the heap's current worst score.
			worst := sc.Heap.Candidates[0]
			if curr.Distance > worst.Score {
				// If the closest candidate in the queue is worse than the worst in our result set,
				// and we have enough results, we can stop?
				// Only if we are sure we won't find anything better.
				// In graph search, we might.
				// But this is a standard pruning heuristic.
				break
			}
		}

		// Neighbors
		r := int(s.header.MaxDegree)
		start := int(curr.Node) * r
		neighbors := s.graph[start : start+r]

		for _, neighbor := range neighbors {
			if neighbor == 0xFFFFFFFF {
				continue
			}

			neighborID := model.RowID(neighbor)
			if !sc.Visited.Visited(neighborID) {
				sc.Visited.Visit(neighborID)
				d := distFn(neighbor)

				// Add to traversal queue
				sc.Candidates.PushItem(searcher.PriorityQueueItem{Node: neighborID, Distance: d})

				// Add to result set
				pushToHeap(neighbor, d)
			}
		}
	}

	return nil
}

func (s *Segment) readMetadata(rowID uint32) (metadata.Document, error) {
	if s.metadataOffsets == nil {
		return nil, nil
	}
	if rowID >= s.header.RowCount {
		return nil, errors.New("rowID out of bounds")
	}

	start := s.metadataOffsets[rowID]
	end := s.metadataOffsets[rowID+1]
	if start == end {
		return nil, nil
	}

	offsetsSize := uint64(s.header.RowCount+1) * 8
	dataStart := s.header.MetadataOffset + offsetsSize

	if dataStart+end > uint64(len(s.data)) {
		return nil, errors.New("metadata blob out of bounds")
	}

	blob := s.data[dataStart+start : dataStart+end]

	var m map[string]interface{}
	if err := json.Unmarshal(blob, &m); err != nil {
		return nil, err
	}

	return metadata.FromMap(m)
}

// Fetch resolves RowIDs to payload columns.
func (s *Segment) Fetch(ctx context.Context, rows []uint32, cols []string) (segment.RecordBatch, error) {
	fetchVectors := cols == nil
	fetchMetadata := cols == nil
	fetchPayloads := cols == nil

	if cols != nil {
		fetchVectors = false
		for _, c := range cols {
			switch c {
			case "vector":
				fetchVectors = true
			case "metadata":
				fetchMetadata = true
			case "payload":
				fetchPayloads = true
			}
		}
	}

	batch := &segment.SimpleRecordBatch{
		PKs: make([]model.PrimaryKey, len(rows)),
	}
	if fetchVectors {
		batch.Vectors = make([][]float32, len(rows))
	}
	if fetchMetadata {
		batch.Metadatas = make([]metadata.Document, len(rows))
	}
	if fetchPayloads {
		batch.Payloads = make([][]byte, len(rows))
	}

	dim := int(s.header.Dim)
	for i, rowID := range rows {
		if rowID >= s.header.RowCount {
			return nil, fmt.Errorf("rowID %d out of bounds", rowID)
		}

		// Fetch PK
		batch.PKs[i] = model.PrimaryKey(s.pks[rowID])

		// Fetch Vector
		if fetchVectors {
			vec := s.vectors[int(rowID)*dim : (int(rowID)+1)*dim]
			v := make([]float32, dim)
			copy(v, vec)
			batch.Vectors[i] = v
		}

		// Fetch Metadata
		if fetchMetadata {
			md, err := s.readMetadata(rowID)
			if err != nil {
				return nil, err
			}
			batch.Metadatas[i] = md
		}

		// Fetch Payload
		if fetchPayloads && s.payloadBlob != nil && uint32(rowID) < s.payloadCount {
			start := s.payloadOffsets[rowID]
			end := s.payloadOffsets[rowID+1]
			size := end - start
			// Offset in file is 4 (count) + (count+1)*8 (offsets) + start
			dataOffset := 4 + uint64(s.payloadCount+1)*8 + start

			p := make([]byte, size)
			if _, err := s.payloadBlob.ReadAt(p, int64(dataOffset)); err != nil {
				return nil, err
			}
			batch.Payloads[i] = p
		}
	}

	return batch, nil
}

// Iterate iterates over all vectors in the segment.
func (s *Segment) Iterate(fn func(rowID uint32, pk model.PrimaryKey, vec []float32, md metadata.Document, payload []byte) error) error {
	dim := int(s.header.Dim)
	for i := 0; i < int(s.header.RowCount); i++ {
		pk := model.PrimaryKey(s.pks[i])
		vec := s.vectors[i*dim : (i+1)*dim]

		var payload []byte
		if s.payloadBlob != nil && i < int(s.payloadCount) {
			start := s.payloadOffsets[i]
			end := s.payloadOffsets[i+1]
			size := end - start
			if size > 0 {
				payload = make([]byte, size)
				// Offset is relative to start of payload data (after header + offsets)
				// Header: 4 bytes
				// Offsets: (count + 1) * 8 bytes
				dataStart := 4 + int64(s.payloadCount+1)*8
				if _, err := s.payloadBlob.ReadAt(payload, dataStart+int64(start)); err != nil {
					return err
				}
			}
		}

		md, err := s.readMetadata(uint32(i))
		if err != nil {
			return err
		}

		if err := fn(uint32(i), pk, vec, md, payload); err != nil {
			return err
		}
	}
	return nil
}

// Rerank computes exact distances for a candidate set.
func (s *Segment) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	for _, c := range cands {
		rowID := uint32(c.Loc.RowID)
		vec, err := s.Get(rowID)
		if err != nil {
			continue
		}
		dist := s.distFunc(q, vec)
		c.Score = dist
		c.Approx = false
		dst = append(dst, c)
	}
	return dst, nil
}

// Size returns the size of the segment in bytes.
func (s *Segment) Size() int64 {
	return int64(len(s.data))
}

// Close closes the segment.
func (s *Segment) Close() error {
	if s.blob != nil {
		return s.blob.Close()
	}
	return nil
}

// Advise hints the kernel about access patterns.
func (s *Segment) Advise(pattern segment.AccessPattern) error {
	// TODO: Expose Advise on Blob interface?
	// For now, we can check if blob is mappable or has Advise method.
	// But blobstore.Blob doesn't have Advise.
	// We can skip it for now or add it to Blob interface.
	return nil
}
