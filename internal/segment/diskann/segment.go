package diskann

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"unsafe"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/hupe1980/vecgo/internal/hash"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
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
	vectors []float32  // Full precision vectors
	ids     []model.ID // Primary keys
	graph   []uint32   // Adjacency list (flat)
	pqCodes []byte     // PQ codes

	// Metadata
	metadataOffsets []uint64
	index           *imetadata.UnifiedIndex

	// Payload
	payloadCount   uint32
	payloadOffsets []uint64

	// Helpers
	pq       *quantization.ProductQuantizer
	rq       *quantization.RaBitQuantizer
	iq       *quantization.Int4Quantizer
	distFunc distance.Func
}

// GetID returns the external ID for a given internal row ID.
func (s *Segment) GetID(ctx context.Context, rowID uint32) (model.ID, bool) {
	if int(rowID) >= int(s.header.RowCount) {
		return 0, false
	}
	if s.ids != nil {
		return s.ids[rowID], true
	}
	return s.readID(ctx, rowID)
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
func Open(ctx context.Context, blob blobstore.Blob, opts ...Option) (*Segment, error) {
	var data []byte
	// Prefer MMap for local files
	if m, ok := blob.(blobstore.Mappable); ok {
		var err error
		data, err = m.Bytes()
		if err != nil {
			return nil, err
		}
	}
	// If not mappable, we default to Lazy Loading (data = nil).
	// We no longer read the entire file into memory.

	s := &Segment{
		blob: blob,
		data: data,
	}

	for _, opt := range opts {
		opt(s)
	}

	if err := s.load(ctx); err != nil {
		s.Close()
		return nil, err
	}

	return s, nil
}

func (s *Segment) load(ctx context.Context) error {
	var err error
	if s.data != nil {
		if len(s.data) < HeaderSize {
			return errors.New("file too small")
		}
		s.header, err = DecodeHeader(s.data)
	} else {
		// Lazy load: read only header
		headerBytes := make([]byte, HeaderSize)
		if _, err := s.blob.ReadAt(ctx, headerBytes, 0); err != nil {
			return fmt.Errorf("failed to read header: %w", err)
		}
		s.header, err = DecodeHeader(headerBytes)
	}
	if err != nil {
		return err
	}

	if s.payloadBlob != nil {
		// Load payload offsets
		// Format: [Count uint32][Offsets uint64...]
		// Read header
		header := make([]byte, 4)
		if _, err := s.payloadBlob.ReadAt(ctx, header, 0); err != nil {
			return fmt.Errorf("failed to read payload header: %w", err)
		}
		count := binary.LittleEndian.Uint32(header)
		s.payloadCount = count

		// Read offsets
		offsetsSize := int(count+1) * 8
		offsetsBytes := make([]byte, offsetsSize)
		if _, err := s.payloadBlob.ReadAt(ctx, offsetsBytes, 4); err != nil {
			return fmt.Errorf("failed to read payload offsets: %w", err)
		}

		s.payloadOffsets = make([]uint64, count+1)
		for i := 0; i < int(count+1); i++ {
			s.payloadOffsets[i] = binary.LittleEndian.Uint64(offsetsBytes[i*8:])
		}
	}

	if s.verifyChecksum && s.header.Checksum != 0 {
		// Verify checksum (CRC32C of body)
		// Only verify if we have data in memory or if explicitly requested (TODO: streaming check)
		if len(s.data) > HeaderSize {
			body := s.data[HeaderSize:]
			sum := hash.CRC32C(body)
			if sum != s.header.Checksum {
				return fmt.Errorf("checksum mismatch: expected %x, got %x", s.header.Checksum, sum)
			}
		}
	}

	// Validate size if data is loaded
	if s.data != nil {
		minSize := s.header.PKOffset + uint64(s.header.RowCount)*8
		if uint64(len(s.data)) < minSize {
			return fmt.Errorf("file size too small: expected at least %d, got %d", minSize, len(s.data))
		}
	}

	// Setup vectors pointer if data is loaded
	if s.data != nil {
		if s.header.VectorOffset+uint64(s.header.RowCount)*uint64(s.header.Dim)*4 > uint64(len(s.data)) {
			return errors.New("vector section out of bounds")
		}
		// Unsafe cast to []float32
		// Note: This assumes little-endian architecture matching file format
		vectorBytes := s.data[s.header.VectorOffset : s.header.VectorOffset+uint64(s.header.RowCount)*uint64(s.header.Dim)*4]
		if len(vectorBytes) > 0 {
			// s.vectors = unsafe.Slice((*float32)(unsafe.Pointer(&vectorBytes[0])), int(s.header.RowCount)*int(s.header.Dim))
			// Use reflect-less unsafe cast for older Go versions compatibility if needed, but unsafe.Slice is best
			s.vectors = unsafe.Slice((*float32)(unsafe.Pointer(&vectorBytes[0])), int(s.header.RowCount)*int(s.header.Dim))
		} else {
			s.vectors = make([]float32, 0)
		}
	}

	// Setup IDs if data is loaded
	// Zero-copy: model.ID is uint64, so we can directly cast the slice
	if s.data != nil {
		idSize := uint64(s.header.RowCount) * 8
		if s.header.PKOffset+idSize > uint64(len(s.data)) {
			return errors.New("ID section out of bounds")
		}
		idBytes := s.data[s.header.PKOffset : s.header.PKOffset+idSize]
		if len(idBytes) > 0 {
			s.ids = unsafe.Slice((*model.ID)(unsafe.Pointer(&idBytes[0])), int(s.header.RowCount))
		} else {
			s.ids = make([]model.ID, 0)
		}
	}

	// Setup Metadata
	if s.header.MetadataOffset > 0 {
		offsetsSize := uint64(s.header.RowCount+1) * 8
		if s.data != nil {
			if s.header.MetadataOffset+offsetsSize > uint64(len(s.data)) {
				return errors.New("metadata offsets section out of bounds")
			}
			offsetsBytes := s.data[s.header.MetadataOffset : s.header.MetadataOffset+offsetsSize]
			s.metadataOffsets = unsafe.Slice((*uint64)(unsafe.Pointer(&offsetsBytes[0])), int(s.header.RowCount)+1)
		}
	}

	// Setup Metadata Index
	if s.header.MetadataIndexOffset > 0 {
		var r io.Reader
		if s.data != nil {
			if s.header.MetadataIndexOffset > uint64(len(s.data)) {
				return errors.New("metadata index offset out of bounds")
			}
			r = bytes.NewReader(s.data[s.header.MetadataIndexOffset:])
		} else {
			// Calculate size of metadata index section
			// It starts at MetadataIndexOffset and goes until... ends or Graph?
			// The layout is usually header -> vectors -> IDs -> Metadata Offsets -> Metadata Index -> Graph -> PQ
			// Let's assume it ends at GraphOffset.
			end := s.header.GraphOffset
			if end == 0 {
				// If no graph (unlikely for diskann), then check PQ?
				// Worst case use file size.
				end = uint64(s.blob.Size())
			}
			size := int64(end - s.header.MetadataIndexOffset)
			r = io.NewSectionReader(blobstore.ReaderAt(s.blob), int64(s.header.MetadataIndexOffset), size)
		}

		s.index = imetadata.NewUnifiedIndex()
		// ReadInvertedIndex might need a ByteReader or similar, io.Reader should be fine
		if err := s.index.ReadInvertedIndex(r); err != nil {
			return fmt.Errorf("failed to read metadata index: %w", err)
		}
		s.index.SetDocumentProvider(func(ctx context.Context, id model.RowID) (metadata.Document, bool) {
			doc, err := s.readMetadata(ctx, uint32(id))
			if err != nil || doc == nil {
				return nil, false
			}
			return doc, true
		})
	}

	// Setup graph pointer
	graphSize := uint64(s.header.RowCount) * uint64(s.header.MaxDegree) * 4
	if s.data != nil {
		if s.header.GraphOffset+graphSize > uint64(len(s.data)) {
			return errors.New("graph section out of bounds")
		}
		graphBytes := s.data[s.header.GraphOffset : s.header.GraphOffset+graphSize]
		if len(graphBytes) > 0 {
			s.graph = unsafe.Slice((*uint32)(unsafe.Pointer(&graphBytes[0])), int(s.header.RowCount)*int(s.header.MaxDegree))
		} else {
			s.graph = make([]uint32, 0)
		}
	}

	// Setup Quantization
	switch quantization.Type(s.header.QuantizationType) {
	case quantization.TypePQ:
		if err := s.loadPQ(ctx); err != nil {
			return err
		}
	case quantization.TypeRaBitQ:
		if err := s.loadRaBitQ(ctx); err != nil {
			return err
		}
	case quantization.TypeINT4:
		if err := s.loadINT4(ctx); err != nil {
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

func (s *Segment) loadPQ(ctx context.Context) error {
	m := int(s.header.PQSubvectors)
	k := int(s.header.PQCentroids)
	subDim := int(s.header.Dim) / m

	// PQ Codes
	pqCodesSize := uint64(s.header.RowCount) * uint64(m)
	if s.data != nil {
		if s.header.PQCodesOffset+pqCodesSize > uint64(len(s.data)) {
			return errors.New("PQ codes section out of bounds")
		}
		s.pqCodes = s.data[s.header.PQCodesOffset : s.header.PQCodesOffset+pqCodesSize]
	}

	// PQ Codebooks
	offset := s.header.PQCodebookOffset

	// Helper to read bytes
	readBytes := func(off uint64, size uint64) ([]byte, error) {
		if s.data != nil {
			if off+size > uint64(len(s.data)) {
				return nil, errors.New("out of bounds")
			}
			return s.data[off : off+size], nil
		}
		buf := make([]byte, size)
		if _, err := s.blob.ReadAt(ctx, buf, int64(off)); err != nil {
			return nil, err
		}
		return buf, nil
	}

	// Scales (M * 4 bytes)
	scalesSize := uint64(m * 4)
	scalesBytes, err := readBytes(offset, scalesSize)
	if err != nil {
		return fmt.Errorf("failed to read PQ scales: %w", err)
	}
	scales := make([]float32, m)
	for i := 0; i < m; i++ {
		bits := binary.LittleEndian.Uint32(scalesBytes[i*4:])
		scales[i] = math.Float32frombits(bits)
	}
	offset += scalesSize

	// Offsets (M * 4 bytes)
	offsetsSize := uint64(m * 4)
	offsetsBytes, err := readBytes(offset, offsetsSize)
	if err != nil {
		return fmt.Errorf("failed to read PQ offsets: %w", err)
	}
	offsets := make([]float32, m)
	for i := 0; i < m; i++ {
		bits := binary.LittleEndian.Uint32(offsetsBytes[i*4:])
		offsets[i] = math.Float32frombits(bits)
	}
	offset += offsetsSize

	// Codebooks (M * K * SubDim * 1 byte)
	codebooksSize := uint64(m * k * subDim)
	cbBytes, err := readBytes(offset, codebooksSize)
	if err != nil {
		return fmt.Errorf("failed to read PQ codebooks: %w", err)
	}
	codebooks := make([]int8, m*k*subDim)
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

func (s *Segment) loadINT4(ctx context.Context) error {
	// Read Params (CodebookOffset)
	if s.header.PQCodebookOffset == 0 {
		return errors.New("missing INT4 params")
	}

	size := int64(s.header.PKOffset - s.header.PQCodebookOffset)
	if size <= 0 {
		return errors.New("invalid INT4 params size")
	}

	params := make([]byte, size)
	if s.data != nil {
		if s.header.PQCodebookOffset+uint64(size) > uint64(len(s.data)) {
			return errors.New("INT4 params out of bounds")
		}
		copy(params, s.data[s.header.PQCodebookOffset:][:size])
	} else {
		if _, err := s.blob.ReadAt(ctx, params, int64(s.header.PQCodebookOffset)); err != nil {
			return err
		}
	}

	s.iq = quantization.NewInt4Quantizer(int(s.header.Dim))
	if err := s.iq.UnmarshalBinary(params); err != nil {
		return err
	}

	// Codes
	if s.header.PQCodesOffset == 0 {
		return errors.New("missing INT4 codes")
	}

	codeSize := (uint64(s.header.Dim) + 1) / 2
	totalCodesSize := codeSize * uint64(s.header.RowCount)

	if s.data != nil {
		if s.header.PQCodesOffset+totalCodesSize > uint64(len(s.data)) {
			return errors.New("INT4 codes out of bounds")
		}
		s.pqCodes = s.data[s.header.PQCodesOffset : s.header.PQCodesOffset+totalCodesSize]
	}
	return nil
}

func (s *Segment) readINT4Code(ctx context.Context, rowID uint32, out []byte) error {
	dim := uint64(s.header.Dim)
	codeSize := (dim + 1) / 2
	if uint64(len(out)) != codeSize {
		return errors.New("buffer size mismatch for INT4 code")
	}
	offset := int64(s.header.PQCodesOffset) + int64(rowID)*int64(codeSize)

	buf, err := s.readBlock(ctx, offset, int(codeSize), cache.CacheKindColumnBlocks)
	if err != nil {
		return err
	}
	copy(out, buf)
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

// HasGraphIndex returns true for DiskANN segments.
// DiskANN uses Vamana graph for approximate nearest neighbor search.
func (s *Segment) HasGraphIndex() bool {
	return true
}

// Get returns the vector for the given ID.
func (s *Segment) Get(ctx context.Context, id uint32) ([]float32, error) {
	if id >= s.header.RowCount {
		return nil, errors.New("id out of bounds")
	}
	dim := int(s.header.Dim)

	if s.vectors != nil {
		start := int(id) * dim
		return s.vectors[start : start+dim], nil
	}

	vec := make([]float32, dim)
	if err := s.readVector(ctx, id, vec); err != nil {
		return nil, err
	}
	return vec, nil
}

// Search performs an approximate nearest neighbor search.
func (s *Segment) Search(ctx context.Context, q []float32, k int, filter segment.Filter, opts model.SearchOptions, searcherCtx *searcher.Searcher) error {
	// DiskANN search list size (L) should be larger than k for proper graph exploration.
	// Default: L = k + 100, ensures reasonable recall even for small k.
	// If RefineFactor > 1.0, use it as a multiplier (allows caller to increase search breadth).
	l := k + 100
	if opts.RefineFactor > 1.0 {
		l = int(float32(k) * opts.RefineFactor)
	}
	// Ensure minimum search breadth
	if l < k+50 {
		l = k + 50
	}

	return s.searchInternal(ctx, q, k, l, filter, opts.Filter, searcherCtx)
}

func (s *Segment) searchInternal(ctx context.Context, query []float32, k int, l int, filter segment.Filter, metadataFilter *metadata.FilterSet, searcherCtx *searcher.Searcher) error {
	// Metadata filter
	var metadataFilterFn func(model.RowID) bool
	if s.index != nil && metadataFilter != nil {
		metadataFilterFn = s.index.CreateStreamingFilter(ctx, metadataFilter)
	}

	// Distance function wrapper
	// For lazy-load paths (not mmap), use IOBuffer from searcherCtx to avoid allocations.
	var distFn func(id uint32) (float32, error)
	if s.rq != nil {
		bytesPerVec := int(((s.header.Dim+63)/64)*8 + 4)
		if s.pqCodes != nil {
			distFn = func(id uint32) (float32, error) {
				start := int(id) * bytesPerVec
				code := s.pqCodes[start : start+bytesPerVec]
				return s.rq.Distance(query, code)
			}
		} else {
			// Lazy load path - use IOBuffer if available
			var code []byte
			if searcherCtx != nil && cap(searcherCtx.IOBuffer) >= bytesPerVec {
				code = searcherCtx.IOBuffer[:bytesPerVec]
			} else {
				code = make([]byte, bytesPerVec)
			}
			distFn = func(id uint32) (float32, error) {
				if err := s.readRaBitQCode(ctx, id, code); err != nil {
					return 0, err
				}
				return s.rq.Distance(query, code)
			}
		}
	} else if s.pq != nil {
		m := int(s.header.PQSubvectors)
		if s.pqCodes != nil {
			distFn = func(id uint32) (float32, error) {
				code := s.pqCodes[int(id)*m : int(id+1)*m]
				return s.pq.ComputeAsymmetricDistance(query, code)
			}
		} else {
			// Lazy load path - use IOBuffer if available
			var code []byte
			if searcherCtx != nil && cap(searcherCtx.IOBuffer) >= m {
				code = searcherCtx.IOBuffer[:m]
			} else {
				code = make([]byte, m)
			}
			distFn = func(id uint32) (float32, error) {
				if err := s.readPQCode(ctx, id, code); err != nil {
					return 0, err
				}
				return s.pq.ComputeAsymmetricDistance(query, code)
			}
		}
	} else if s.iq != nil {
		bytesPerVec := (int(s.header.Dim) + 1) / 2
		if s.pqCodes != nil {
			distFn = func(id uint32) (float32, error) {
				start := int(id) * bytesPerVec
				code := s.pqCodes[start : start+bytesPerVec]
				return s.iq.L2Distance(query, code)
			}
		} else {
			// Lazy load path - use IOBuffer if available
			var code []byte
			if searcherCtx != nil && cap(searcherCtx.IOBuffer) >= bytesPerVec {
				code = searcherCtx.IOBuffer[:bytesPerVec]
			} else {
				code = make([]byte, bytesPerVec)
			}
			distFn = func(id uint32) (float32, error) {
				if err := s.readINT4Code(ctx, id, code); err != nil {
					return 0, err
				}
				return s.iq.L2Distance(query, code)
			}
		}
	} else {
		distFn = func(id uint32) (float32, error) {
			vec, err := s.Get(ctx, id)
			if err != nil {
				return 0, err
			}
			return s.distFunc(query, vec), nil
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
	sc.ScratchCandidates.Reset() // MinHeap for traversal (explore closest first)
	// sc.Heap is already initialized by caller (engine) or above

	// Greedy search
	startNode := s.header.Entrypoint

	sc.Visited.Visit(model.RowID(startNode))
	startDist, err := distFn(startNode)
	if err != nil {
		return err
	}

	sc.ScratchCandidates.PushItem(searcher.PriorityQueueItem{Node: model.RowID(startNode), Distance: startDist})

	// Helper to push to heap
	pushToHeap := func(id uint32, dist float32) {
		if filter != nil && !filter.Matches(id) {
			return
		}
		if metadataFilterFn != nil && !metadataFilterFn(model.RowID(id)) {
			return
		}

		c := searcher.InternalCandidate{SegmentID: uint32(s.ID()), RowID: id, Score: dist, Approx: true}
		if sc.Heap.Len() < k {
			sc.Heap.Push(c)
		} else {
			top := sc.Heap.Candidates[0]
			if searcher.InternalCandidateBetter(c, top, sc.Heap.Descending()) {
				sc.Heap.ReplaceTop(c)
			}
		}
	}

	// Reset heap for local search if we were not passed a context?
	// No, if we created it locally, we reset it.
	// If passed from engine, it accumulates.

	// DiskANN search uses greedy beam search:
	// - sc.ScratchCandidates (MinHeap): traversal queue, explore closest candidates first
	// - sc.Heap (MaxHeap): result set, keep top-K smallest distances

	pushToHeap(startNode, startDist)

	// Context check interval counter
	iterations := 0

	for sc.ScratchCandidates.Len() > 0 {
		// Periodic context check (every 128 iterations)
		iterations++
		if iterations&127 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		// Pop closest unexpanded (MinHeap returns smallest distance first)
		curr, _ := sc.ScratchCandidates.PopItem()

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
		var neighbors []uint32
		if s.graph != nil {
			r := int(s.header.MaxDegree)
			start := int(curr.Node) * r
			neighbors = s.graph[start : start+r]
		} else {
			var err error
			neighbors, err = s.readGraphNode(ctx, uint32(curr.Node))
			if err != nil {
				return err
			}
		}

		for _, neighbor := range neighbors {
			if neighbor == 0xFFFFFFFF {
				continue
			}

			neighborID := model.RowID(neighbor)
			if !sc.Visited.Visited(neighborID) {
				sc.Visited.Visit(neighborID)
				d, err := distFn(neighbor)
				if err != nil {
					return err
				}

				// Add to traversal queue (MinHeap)
				sc.ScratchCandidates.PushItem(searcher.PriorityQueueItem{Node: neighborID, Distance: d})

				// Add to result set
				pushToHeap(neighbor, d)
			}
		}
	}

	return nil
}

func (s *Segment) readMetadata(ctx context.Context, rowID uint32) (metadata.Document, error) {
	if s.header.MetadataOffset == 0 {
		return nil, nil
	}
	if rowID >= s.header.RowCount {
		return nil, errors.New("rowID out of bounds")
	}

	var start, end uint64
	if s.metadataOffsets != nil {
		start = s.metadataOffsets[rowID]
		end = s.metadataOffsets[rowID+1]
	} else {
		// Read offsets from blob
		offsetPos := int64(s.header.MetadataOffset) + int64(rowID)*8
		buf := make([]byte, 16)
		if _, err := s.blob.ReadAt(ctx, buf, offsetPos); err != nil {
			return nil, err
		}
		start = binary.LittleEndian.Uint64(buf[0:8])
		end = binary.LittleEndian.Uint64(buf[8:16])
	}

	if start == end {
		return nil, nil
	}

	offsetsSize := uint64(s.header.RowCount+1) * 8
	dataStart := s.header.MetadataOffset + offsetsSize

	var blob []byte
	if s.data != nil {
		if dataStart+end > uint64(len(s.data)) {
			return nil, errors.New("metadata blob out of bounds")
		}
		blob = s.data[dataStart+start : dataStart+end]
	} else {
		blob = make([]byte, end-start)
		if _, err := s.blob.ReadAt(ctx, blob, int64(dataStart+start)); err != nil {
			return nil, err
		}
	}

	var md metadata.Document
	if err := md.UnmarshalBinary(blob); err != nil {
		return nil, err
	}
	return md, nil
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
		IDs: make([]model.ID, len(rows)),
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

	// Optimization: Allocate single backing array for vectors to reduce GC pressure
	var vectorBacking []float32
	dim := int(s.header.Dim)
	if fetchVectors {
		vectorBacking = make([]float32, len(rows)*dim)
	}

	for i, rowID := range rows {
		// Periodic context check (every 64 rows)
		if i&63 == 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
		}

		if rowID >= s.header.RowCount {
			return nil, fmt.Errorf("rowID %d out of bounds", rowID)
		}

		// Fetch ID
		if id, ok := s.GetID(ctx, rowID); ok {
			batch.IDs[i] = id
		} else {
			return nil, fmt.Errorf("failed to get ID for row %d", rowID)
		}

		// Fetch Vector
		if fetchVectors {
			vec, err := s.Get(ctx, rowID)
			if err != nil {
				return nil, err
			}
			// Copy into backing array
			dst := vectorBacking[i*dim : (i+1)*dim]
			copy(dst, vec)
			batch.Vectors[i] = dst
		}

		// Fetch Metadata
		if fetchMetadata {
			md, err := s.readMetadata(ctx, rowID)
			if err != nil {
				return nil, err
			}
			batch.Metadatas[i] = md
		}

		// Fetch Payload
		if fetchPayloads && s.payloadBlob != nil && rowID < s.payloadCount {
			start := s.payloadOffsets[rowID]
			end := s.payloadOffsets[rowID+1]
			size := end - start
			// Offset in file is 4 (count) + (count+1)*8 (offsets) + start
			dataOffset := 4 + uint64(s.payloadCount+1)*8 + start

			p := make([]byte, size)
			if _, err := s.payloadBlob.ReadAt(ctx, p, int64(dataOffset)); err != nil {
				return nil, err
			}
			batch.Payloads[i] = p
		}
	}

	return batch, nil
}

func (s *Segment) FetchIDs(ctx context.Context, rows []uint32, dst []model.ID) error {
	if len(dst) != len(rows) {
		return fmt.Errorf("dst length mismatch")
	}
	for i, rowID := range rows {
		// Periodic context check (every 256 rows)
		if i&255 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		if rowID >= s.header.RowCount {
			return fmt.Errorf("rowID %d out of bounds", rowID)
		}
		if id, ok := s.GetID(ctx, rowID); ok {
			dst[i] = id
		} else {
			return fmt.Errorf("failed to get ID for row %d", rowID)
		}
	}
	return nil
}

// FetchVectorsInto copies vectors for the given rows into dst.
// dst must have len >= len(rows)*dim.
// All vectors are always valid for diskann segment (no deletions at segment level).
func (s *Segment) FetchVectorsInto(ctx context.Context, rows []uint32, dim int, dst []float32) ([]bool, error) {
	if dim != int(s.header.Dim) {
		return nil, fmt.Errorf("dimension mismatch: got %d, expected %d", dim, s.header.Dim)
	}
	if len(dst) < len(rows)*dim {
		return nil, fmt.Errorf("dst too small: need %d, got %d", len(rows)*dim, len(dst))
	}

	for i, rowID := range rows {
		// Periodic context check (every 256 rows)
		if i&255 == 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
		}

		if rowID >= s.header.RowCount {
			return nil, fmt.Errorf("rowID %d out of bounds", rowID)
		}

		src := s.vectors[int(rowID)*dim : int(rowID+1)*dim]
		dstSlice := dst[i*dim : (i+1)*dim]
		copy(dstSlice, src)
	}

	return nil, nil // All valid
}

// FetchVectorDirect returns a slice pointing directly to the mmap'd vector data.
// This is zero-copy and very fast, but:
// - The returned slice must NOT be modified
// - The slice is only valid while the segment is open
// - The slice points to mmap'd memory
// Returns nil if the rowID is out of bounds.
func (s *Segment) FetchVectorDirect(rowID uint32) []float32 {
	if rowID >= s.header.RowCount {
		return nil
	}
	dim := int(s.header.Dim)
	return s.vectors[int(rowID)*dim : int(rowID+1)*dim]
}

// Iterate iterates over all vectors in the segment.
// The context is used for cancellation during long iterations.
func (s *Segment) Iterate(ctx context.Context, fn func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error) error {
	for i := 0; i < int(s.header.RowCount); i++ {
		// Periodic context check (every 256 rows)
		if i&255 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		id, ok := s.GetID(ctx, uint32(i))
		if !ok {
			return fmt.Errorf("failed to get ID for row %d", i)
		}
		vec, err := s.Get(ctx, uint32(i))
		if err != nil {
			return err
		}

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
				if _, err := s.payloadBlob.ReadAt(ctx, payload, dataStart+int64(start)); err != nil {
					return err
				}
			}
		}

		md, err := s.readMetadata(ctx, uint32(i))
		if err != nil {
			return err
		}

		if err := fn(uint32(i), id, vec, md, payload); err != nil {
			return err
		}
	}
	return nil
}

// Rerank computes exact distances for a candidate set.
func (s *Segment) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	for i, c := range cands {
		// Periodic context check (every 64 candidates)
		if i&63 == 0 {
			select {
			case <-ctx.Done():
				return dst, ctx.Err()
			default:
			}
		}

		rowID := uint32(c.Loc.RowID)
		vec, err := s.Get(ctx, rowID)
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
	if s.data != nil {
		return int64(len(s.data))
	}
	return s.blob.Size()
}

// Close closes the segment.
func (s *Segment) Close() error {
	var errs []error
	if s.payloadBlob != nil {
		if err := s.payloadBlob.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if s.blob != nil {
		if err := s.blob.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Advise hints the kernel about access patterns.
func (s *Segment) Advise(pattern segment.AccessPattern) error {
	// TODO: Expose Advise on Blob interface?
	// For now, we can check if blob is mappable or has Advise method.
	// But blobstore.Blob doesn't have Advise.
	// We can skip it for now or add it to Blob interface.
	return nil
}

// readBlock reads a aligned block-based chunk from the blob using the cache.
func (s *Segment) readBlock(ctx context.Context, offset int64, size int, kind cache.CacheKind) ([]byte, error) {
	if s.cache == nil {
		buf := make([]byte, size)
		if _, err := s.blob.ReadAt(ctx, buf, offset); err != nil {
			return nil, err
		}
		return buf, nil
	}

	const pageSize = 4096
	startPage := offset / pageSize
	endPage := (offset + int64(size) - 1) / pageSize

	totalBuf := make([]byte, size)

	for page := startPage; page <= endPage; page++ {
		key := cache.CacheKey{
			Kind:      kind,
			SegmentID: s.ID(),
			Offset:    uint64(page),
		}

		var pageData []byte
		var ok bool

		pageData, ok = s.cache.Get(ctx, key)
		if !ok {
			// Read aligned page
			pageStart := page * pageSize
			readBuf := make([]byte, pageSize)
			n, err := s.blob.ReadAt(ctx, readBuf, pageStart)
			if err != nil && !errors.Is(err, io.EOF) {
				return nil, err
			}
			if n == 0 {
				return nil, io.EOF
			}
			if n < pageSize {
				readBuf = readBuf[:n]
			}
			pageData = readBuf
			s.cache.Set(ctx, key, pageData)
		}

		// Calculate overlap
		pageStart := page * pageSize

		// Intersection of [offset, offset+size) and [pageStart, pageStart+len(pageData))

		// Start index in pageData
		srcStart := offset - pageStart
		if srcStart < 0 {
			srcStart = 0
		}

		// End index in pageData
		srcEnd := (offset + int64(size)) - pageStart
		if srcEnd > int64(len(pageData)) {
			srcEnd = int64(len(pageData))
		}

		if srcStart < srcEnd {
			// Start index in totalBuf
			dstStart := (pageStart + srcStart) - offset
			copy(totalBuf[dstStart:], pageData[srcStart:srcEnd])
		}
	}

	return totalBuf, nil
}

// EvaluateFilter returns a bitmap of rows matching the filter.
func (s *Segment) EvaluateFilter(ctx context.Context, filter *metadata.FilterSet) (segment.Bitmap, error) {
	if s.index == nil {
		// If no index, we can't efficiently evaluate.
		// Fallback should be handled by caller (full scan or error).
		// For L1 segments, we expect an index.
		return nil, errors.New("metadata index not available")
	}
	return s.index.Query(filter)
}

// EvaluateFilterResult returns filter results using the zero-alloc FilterResult type.
// Uses QueryScratch for scratch space (zero allocations in steady state).
func (s *Segment) EvaluateFilterResult(ctx context.Context, filter *metadata.FilterSet, qs *imetadata.QueryScratch) (imetadata.FilterResult, error) {
	if s.index == nil {
		return imetadata.EmptyResult(), errors.New("metadata index not available")
	}
	return s.index.EvaluateFilterResult(filter, qs), nil
}

// FilterCursor returns a push-based cursor for filtered iteration.
// This is the zero-allocation hot path that avoids Roaring bitmap operations.
func (s *Segment) FilterCursor(filter *metadata.FilterSet) imetadata.FilterCursor {
	if filter == nil || len(filter.Filters) == 0 {
		return imetadata.NewAllCursor(s.header.RowCount)
	}

	if s.index == nil {
		// No index = no way to evaluate efficiently
		return imetadata.GetEmptyCursor()
	}

	// Note: Lock is acquired lazily in each method to avoid deadlock
	// when cursor is inspected (IsEmpty/IsAll) but ForEach is not called
	return &diskannSegmentCursor{
		index:    s.index,
		filter:   filter,
		rowCount: s.header.RowCount,
	}
}

// diskannSegmentCursor wraps UnifiedIndex cursor with lazy lock management.
// The inner cursor is created on first use and cached for subsequent calls.
// Lock is acquired per-method to avoid deadlock when cursor is inspected
// but ForEach is not called.
type diskannSegmentCursor struct {
	index    *imetadata.UnifiedIndex
	filter   *metadata.FilterSet
	rowCount uint32

	// Cached inner cursor (created on first use under lock)
	inner imetadata.FilterCursor
	once  sync.Once
}

// ensureInner creates the inner cursor under lock on first call.
// Subsequent calls return immediately (sync.Once guarantees single init).
func (c *diskannSegmentCursor) ensureInner() {
	c.once.Do(func() {
		c.index.RLock()
		defer c.index.RUnlock()
		c.inner = c.index.FilterCursor(c.filter, c.rowCount)
	})
}

func (c *diskannSegmentCursor) ForEach(fn func(rowID uint32) bool) {
	c.ensureInner()
	c.inner.ForEach(fn)
}

func (c *diskannSegmentCursor) EstimateCardinality() int {
	c.ensureInner()
	return c.inner.EstimateCardinality()
}

func (c *diskannSegmentCursor) IsEmpty() bool {
	c.ensureInner()
	return c.inner.IsEmpty()
}

func (c *diskannSegmentCursor) IsAll() bool {
	c.ensureInner()
	return c.inner.IsAll()
}

// readID reads the ID for the given row from the blob.
func (s *Segment) readID(ctx context.Context, rowID uint32) (model.ID, bool) {
	if s.header.PKOffset == 0 {
		return 0, false
	}
	offset := int64(s.header.PKOffset) + int64(rowID)*8

	buf, err := s.readBlock(ctx, offset, 8, cache.CacheKindColumnBlocks)
	if err != nil {
		return 0, false
	}
	return model.ID(binary.LittleEndian.Uint64(buf)), true
}

// readVector reads the vector for the given row from the blob.
func (s *Segment) readVector(ctx context.Context, rowID uint32, out []float32) error {
	dim := int(s.header.Dim)
	if len(out) != dim {
		return errors.New("output buffer size mismatch")
	}
	offset := int64(s.header.VectorOffset) + int64(rowID)*int64(dim)*4

	buf, err := s.readBlock(ctx, offset, dim*4, cache.CacheKindColumnBlocks)
	if err != nil {
		return err
	}

	// Zero-copy conversion: readBlock returns a properly aligned buffer.
	// Go's allocator aligns to 8 bytes, float32 needs 4-byte alignment.
	src := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), dim)
	copy(out, src)
	return nil
}

// readPQCode reads the PQ code for the given row.
func (s *Segment) readPQCode(ctx context.Context, rowID uint32, out []byte) error {
	m := int(s.header.PQSubvectors)
	if len(out) != m {
		return errors.New("output buffer size mismatch")
	}
	offset := int64(s.header.PQCodesOffset) + int64(rowID)*int64(m)

	buf, err := s.readBlock(ctx, offset, m, cache.CacheKindColumnBlocks)
	if err != nil {
		return err
	}
	copy(out, buf)
	return nil
}

// readGraphNode reads the neighbors for the given row.
func (s *Segment) readGraphNode(ctx context.Context, rowID uint32) ([]uint32, error) {
	maxDegree := int(s.header.MaxDegree)
	offset := int64(s.header.GraphOffset) + int64(rowID)*int64(maxDegree)*4

	buf, err := s.readBlock(ctx, offset, maxDegree*4, cache.CacheKindGraph)
	if err != nil {
		return nil, err
	}

	// Zero-copy conversion: readBlock returns a properly aligned buffer.
	// Go's allocator aligns to 8 bytes, uint32 needs 4-byte alignment.
	neighbors := make([]uint32, maxDegree)
	src := unsafe.Slice((*uint32)(unsafe.Pointer(&buf[0])), maxDegree)
	copy(neighbors, src)
	return neighbors, nil
}

func (s *Segment) loadRaBitQ(_ context.Context) error {
	s.rq = quantization.NewRaBitQuantizer(int(s.header.Dim))

	// Calculate size: (Dim/64)*8 + 4 bytes per vector
	bytesPerVec := uint64(((int(s.header.Dim)+63)/64)*8 + 4)
	size := uint64(s.header.RowCount) * bytesPerVec
	offset := s.header.BQCodesOffset

	if s.data != nil {
		if offset+size > uint64(len(s.data)) {
			return errors.New("RaBitQ codes section out of bounds")
		}
		s.pqCodes = s.data[offset : offset+size]
	}
	return nil
}

// readRaBitQCode reads the RaBitQ code for the given row.
func (s *Segment) readRaBitQCode(ctx context.Context, rowID uint32, out []byte) error {
	size := int(((s.header.Dim+63)/64)*8 + 4)
	if len(out) != size {
		return errors.New("output buffer size mismatch for RaBitQ")
	}
	offset := int64(s.header.BQCodesOffset) + int64(rowID)*int64(size)

	buf, err := s.readBlock(ctx, offset, size, cache.CacheKindColumnBlocks)
	if err != nil {
		return err
	}
	copy(out, buf)
	return nil
}
