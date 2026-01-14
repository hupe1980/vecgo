package flat

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/cache"
	"github.com/hupe1980/vecgo/internal/hash"
	"github.com/hupe1980/vecgo/internal/kmeans"
	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/searcher"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// Segment implements an immutable flat segment.
type Segment struct {
	header *FileHeader
	blob   blobstore.Blob
	data   []byte // Mmapped data or in-memory copy

	// Pointers to data sections
	vectors []float32
	ids     []model.ID

	// Partitioning
	centroids        []float32
	partitionOffsets []uint32
	numPartitions    int

	// Quantization
	sq    *quantization.ScalarQuantizer
	pq    *quantization.ProductQuantizer
	codes []byte

	// Metadata
	metadataOffsets []uint32
	metadataBlob    []byte
	metadataIndex   *imetadata.UnifiedIndex // Inverted index for fast filtering

	// Payload
	payloadBlob    blobstore.Blob
	payloadOffsets []uint64
	payloadCount   uint32

	// Block Stats
	blockStats []BlockStats

	cache          cache.BlockCache
	verifyChecksum bool
	indexMetadata  bool // Build inverted index at open time
}

// GetID returns the external ID for a given internal row ID.
func (s *Segment) GetID(_ context.Context, rowID uint32) (model.ID, bool) {
	if int(rowID) >= len(s.ids) {
		return 0, false
	}
	return s.ids[rowID], true
}

// Option defines a configuration option for the Segment.
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

// WithMetadataIndex enables building an inverted index at segment open time.
// This significantly speeds up filtered search at the cost of memory.
// Recommended for segments that will be searched with filters.
func WithMetadataIndex(index bool) Option {
	return func(s *Segment) {
		s.indexMetadata = index
	}
}

// Open opens a flat segment from a blob.
func Open(ctx context.Context, blob blobstore.Blob, opts ...Option) (*Segment, error) {
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
		if _, err := blob.ReadAt(ctx, data, 0); err != nil {
			return nil, err
		}
	}

	// Parse Header
	header, err := DecodeHeader(data)
	if err != nil {
		blob.Close()
		return nil, err
	}

	s := &Segment{
		header:        header,
		blob:          blob,
		data:          data,
		numPartitions: int(header.NumPartitions),
	}

	for _, opt := range opts {
		opt(s)
	}

	if s.payloadBlob != nil {
		// Load payload offsets
		// Format: [Count uint32][Offsets uint64...]
		// Read header
		header := make([]byte, 4)
		if _, err := s.payloadBlob.ReadAt(ctx, header, 0); err != nil {
			blob.Close()
			s.payloadBlob.Close()
			return nil, fmt.Errorf("failed to read payload header: %w", err)
		}
		count := binary.LittleEndian.Uint32(header)
		s.payloadCount = count

		// Read offsets
		offsetsSize := int(count+1) * 8
		offsetsBytes := make([]byte, offsetsSize)
		if _, err := s.payloadBlob.ReadAt(ctx, offsetsBytes, 4); err != nil {
			blob.Close()
			s.payloadBlob.Close()
			return nil, fmt.Errorf("failed to read payload offsets: %w", err)
		}
		// Copy offsets to avoid unsafe pointer to stack/temp slice if we want to be safe,
		// but here offsetsBytes is allocated on heap.
		// However, we need []uint64.
		s.payloadOffsets = make([]uint64, count+1)
		for i := 0; i < int(count+1); i++ {
			s.payloadOffsets[i] = binary.LittleEndian.Uint64(offsetsBytes[i*8:])
		}
	}

	if s.verifyChecksum && header.Checksum != 0 {
		// Verify checksum (CRC32C of body)
		// Body starts after HeaderSize
		if len(data) > HeaderSize {
			body := data[HeaderSize:]
			sum := hash.CRC32C(body)
			if sum != header.Checksum {
				blob.Close()
				return nil, fmt.Errorf("checksum mismatch: expected %x, got %x", header.Checksum, sum)
			}
		}
	}

	// Set up views
	// Safety: We assume the file is immutable and not truncated while open.
	// We use unsafe to cast []byte to []float32 and []uint64 to avoid copying.

	// Centroids
	if s.numPartitions > 0 {
		cStart := header.CentroidOffset
		cBytes := uint64(s.numPartitions) * uint64(header.Dim) * 4
		if uint64(len(data)) < cStart+cBytes {
			blob.Close()
			return nil, errors.New("file too short for centroids")
		}
		cData := data[cStart : cStart+cBytes]
		s.centroids = unsafe.Slice((*float32)(unsafe.Pointer(&cData[0])), len(cData)/4)

		pStart := header.PartitionOffsetOffset
		pBytes := uint64(s.numPartitions+1) * 4
		if uint64(len(data)) < pStart+pBytes {
			blob.Close()
			return nil, errors.New("file too short for partition offsets")
		}
		pData := data[pStart : pStart+pBytes]
		s.partitionOffsets = unsafe.Slice((*uint32)(unsafe.Pointer(&pData[0])), len(pData)/4)
	}

	// Quantization
	if header.QuantizationType == QuantizationSQ8 {
		qStart := header.QuantizationOffset
		// Mins (dim * 4) + Maxs (dim * 4)
		qBytes := uint64(header.Dim) * 4 * 2
		if uint64(len(data)) < qStart+qBytes {
			blob.Close()
			return nil, errors.New("file too short for quantization metadata")
		}

		qData := data[qStart : qStart+qBytes]
		mins := unsafe.Slice((*float32)(unsafe.Pointer(&qData[0])), header.Dim)
		maxs := unsafe.Slice((*float32)(unsafe.Pointer(&qData[uint64(header.Dim)*4])), header.Dim)

		s.sq = quantization.NewScalarQuantizer(int(header.Dim))
		if err := s.sq.SetBounds(mins, maxs); err != nil {
			blob.Close()
			return nil, err
		}

		cStart := header.CodesOffset
		cBytes := uint64(header.RowCount) * uint64(header.Dim)
		if uint64(len(data)) < cStart+cBytes {
			blob.Close()
			return nil, errors.New("file too short for codes")
		}
		s.codes = data[cStart : cStart+cBytes]
	} else if header.QuantizationType == QuantizationPQ {
		qStart := header.QuantizationOffset
		// m (4) + k (4)
		if uint64(len(data)) < qStart+8 {
			blob.Close()
			return nil, errors.New("file too short for PQ metadata")
		}
		m := int(binary.LittleEndian.Uint32(data[qStart:]))
		k := int(binary.LittleEndian.Uint32(data[qStart+4:]))

		// scales (m*4) + offsets (m*4) + codebooks (m*k*dsub*1)
		dsub := int(header.Dim) / m
		codebookSize := m * k * dsub
		metaSize := 8 + m*4 + m*4 + codebookSize

		if uint64(len(data)) < qStart+uint64(metaSize) {
			blob.Close()
			return nil, errors.New("file too short for PQ metadata")
		}

		scalesData := data[qStart+8 : qStart+8+uint64(m)*4]
		scales := unsafe.Slice((*float32)(unsafe.Pointer(&scalesData[0])), m)

		offsetsData := data[qStart+8+uint64(m)*4 : qStart+8+uint64(m)*8]
		offsets := unsafe.Slice((*float32)(unsafe.Pointer(&offsetsData[0])), m)

		codebooksData := data[qStart+8+uint64(m)*8 : qStart+uint64(metaSize)]
		// Copy codebooks because we might need to cast to int8 and unsafe.Slice on []byte to []int8 is tricky?
		// Actually unsafe.Slice works fine.
		codebooks := unsafe.Slice((*int8)(unsafe.Pointer(&codebooksData[0])), len(codebooksData))

		var err error
		s.pq, err = quantization.NewProductQuantizer(int(header.Dim), m, k)
		if err != nil {
			blob.Close()
			return nil, err
		}
		s.pq.SetCodebooks(codebooks, scales, offsets)

		cStart := header.CodesOffset
		cBytes := uint64(header.RowCount) * uint64(m)
		if uint64(len(data)) < cStart+cBytes {
			blob.Close()
			return nil, errors.New("file too short for codes")
		}
		s.codes = data[cStart : cStart+cBytes]
	}

	// Vectors
	vecStart := header.VectorOffset
	vecBytes := uint64(header.RowCount) * uint64(header.Dim) * 4
	if uint64(len(data)) < vecStart+vecBytes {
		blob.Close()
		return nil, errors.New("file too short for vectors")
	}

	// unsafe cast for vectors
	// Note: This requires the data to be aligned. Mmap usually gives page-aligned data.
	// float32 requires 4-byte alignment.
	vecData := data[vecStart : vecStart+vecBytes]
	if len(vecData) > 0 {
		s.vectors = unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), len(vecData)/4)
	}

	// PKs
	// IDs
	pkStart := header.PKOffset
	pkSize := header.MetadataOffset - header.PKOffset
	if uint64(len(data)) < pkStart+pkSize {
		blob.Close()
		return nil, errors.New("file too short for IDs")
	}

	if header.RowCount > 0 {
		expectedSize := uint64(header.RowCount) * 8
		if pkSize < expectedSize {
			blob.Close()
			return nil, errors.New("id section too small")
		}

		idData := data[pkStart : pkStart+expectedSize]
		// model.ID is uint64, so we can use unsafe.Slice directly
		s.ids = unsafe.Slice((*model.ID)(unsafe.Pointer(&idData[0])), header.RowCount)
	}

	// Metadata
	if header.MetadataOffset > 0 && header.RowCount > 0 {
		mdStart := header.MetadataOffset
		// Offsets: (RowCount + 1) * 4
		offsetBytes := uint64(header.RowCount+1) * 4
		if uint64(len(data)) < mdStart+offsetBytes {
			blob.Close()
			return nil, errors.New("file too short for metadata offsets")
		}

		offsetData := data[mdStart : mdStart+offsetBytes]
		s.metadataOffsets = unsafe.Slice((*uint32)(unsafe.Pointer(&offsetData[0])), len(offsetData)/4)

		// Data blob
		blobStart := mdStart + offsetBytes
		blobSize := s.metadataOffsets[header.RowCount] // Last offset is total size

		if uint64(len(data)) < blobStart+uint64(blobSize) {
			blob.Close()
			return nil, errors.New("file too short for metadata blob")
		}
		s.metadataBlob = data[blobStart : blobStart+uint64(blobSize)]
	}

	// Block Stats
	if header.BlockStatsOffset > 0 {
		bsStart := header.BlockStatsOffset
		if uint64(len(data)) <= bsStart {
			blob.Close()
			return nil, errors.New("file too short for block stats")
		}
		bsData := data[bsStart:]
		if len(bsData) > 0 {
			count, n := binary.Uvarint(bsData)
			if n <= 0 {
				blob.Close()
				return nil, errors.New("invalid block stats count")
			}
			bsData = bsData[n:]
			s.blockStats = make([]BlockStats, count)
			for i := 0; i < int(count); i++ {
				l, n := binary.Uvarint(bsData)
				if n <= 0 {
					blob.Close()
					return nil, errors.New("invalid block stats length")
				}
				bsData = bsData[n:]
				if uint64(len(bsData)) < l {
					blob.Close()
					return nil, errors.New("block stats too short")
				}
				if err := s.blockStats[i].UnmarshalBinary(bsData[:l]); err != nil {
					blob.Close()
					return nil, err
				}
				bsData = bsData[l:]
			}
		}
	}

	// Build metadata inverted index if enabled
	if s.indexMetadata && len(s.metadataOffsets) > 0 {
		s.metadataIndex = imetadata.NewUnifiedIndex()
		for i := 0; i < int(s.header.RowCount); i++ {
			start := s.metadataOffsets[i]
			end := s.metadataOffsets[i+1]
			if start < end {
				mdBytes := s.metadataBlob[start:end]
				var md metadata.Document
				if err := md.UnmarshalBinary(mdBytes); err == nil && md != nil {
					s.metadataIndex.AddInvertedIndex(model.RowID(i), md)
				}
			}
		}
	}

	return s, nil
}

// Size returns the size of the segment in bytes.
func (s *Segment) Size() int64 {
	return int64(len(s.data))
}

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

func (s *Segment) ID() model.SegmentID {
	return model.SegmentID(s.header.SegmentID)
}

// QuantizationType returns the quantization type of the segment.
func (s *Segment) QuantizationType() int {
	return int(s.header.QuantizationType)
}

func (s *Segment) RowCount() uint32 {
	return s.header.RowCount
}

func (s *Segment) Metric() distance.Metric {
	return distance.Metric(s.header.Metric)
}

// Search performs an exact scan.
func (s *Segment) Search(ctx context.Context, q []float32, k int, filter segment.Filter, opts model.SearchOptions, searcherCtx *searcher.Searcher) error {
	descending := s.Metric() != distance.MetricL2

	var h *searcher.CandidateHeap
	if searcherCtx != nil {
		h = searcherCtx.Heap
		// Do NOT reset if we are accumulating results from multiple segments
		// But wait, engine.Search calls Reset() before loop.
		// If we are in a loop, we want to KEEP existing items.
		// So we should NOT Reset here if searcherCtx is provided.
		// But if searcherCtx is nil (legacy/test), we need a new heap.
	} else {
		h = searcher.NewCandidateHeap(k, descending)
	}

	dim := int(s.header.Dim)

	// Precompute PQ table if needed
	var pqTable []float32
	if s.pq != nil {
		var err error
		pqTable, err = s.pq.BuildDistanceTable(q)
		if err != nil {
			return err
		}
	}

	// Scratch buffer for SQ8 batch processing
	var batchScores []float32
	const batchSize = 256
	if s.sq != nil && s.Metric() == distance.MetricL2 {
		if searcherCtx != nil {
			if cap(searcherCtx.Scores) < batchSize {
				searcherCtx.Scores = make([]float32, batchSize)
			}
			batchScores = searcherCtx.Scores[:batchSize]
		} else {
			batchScores = make([]float32, batchSize)
		}
	}

	scan := func(start, end int) error {
		// Check context cancellation periodically (every ~1000 iterations)
		checkInterval := 1024
		lastCheck := start

		// Helper to check context
		checkCtx := func(i int) error {
			if i-lastCheck >= checkInterval {
				lastCheck = i
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
			}
			return nil
		}

		// Optimization for SQ8 L2
		if s.sq != nil && s.Metric() == distance.MetricL2 {
			for i := start; i < end; {
				// Periodic context check
				if err := checkCtx(i); err != nil {
					return err
				}
				// Block skipping
				if i%BlockSize == 0 && i+BlockSize <= end {
					blockIdx := i / BlockSize
					if blockIdx < len(s.blockStats) {
						stats := s.blockStats[blockIdx].Fields
						if filter != nil && !filter.MatchesBlock(stats) {
							i += BlockSize
							continue
						}
						if opts.Filter != nil {
							fs := opts.Filter
							if !matchesFilterSet(fs, stats) {
								i += BlockSize
								continue
							}
						}
					}
				}

				limit := i + batchSize
				if limit > end {
					limit = end
				}
				count := limit - i

				batchCodes := s.codes[i*dim : limit*dim]
				if err := s.sq.L2DistanceBatch(q, batchCodes, count, batchScores[:count]); err != nil {
					return err
				}

				for j := 0; j < count; j++ {
					idx := i + j
					if filter != nil && !filter.Matches(uint32(idx)) {
						continue
					}

					if opts.Filter != nil {
						if len(s.metadataOffsets) > 0 {
							startOffset := s.metadataOffsets[idx]
							endOffset := s.metadataOffsets[idx+1]
							if startOffset < endOffset {
								mdBytes := s.metadataBlob[startOffset:endOffset]
								match, err := opts.Filter.MatchesBinary(mdBytes)
								if err != nil {
									continue
								}
								if !match {
									continue
								}
							} else {
								continue
							}
						}
					}

					cand := searcher.InternalCandidate{
						SegmentID: uint32(s.ID()),
						RowID:     uint32(idx),
						Score:     batchScores[j],
						Approx:    true,
					}

					if h.Len() < k {
						h.Push(cand)
					} else {
						top := h.Candidates[0]
						if searcher.InternalCandidateBetter(cand, top, descending) {
							h.ReplaceTop(cand)
						}
					}
				}
				i += count
			}
			return nil
		}

		for i := start; i < end; {
			// Periodic context check
			if err := checkCtx(i); err != nil {
				return err
			}

			// Block skipping
			if i%BlockSize == 0 && i+BlockSize <= end {
				blockIdx := i / BlockSize
				if blockIdx < len(s.blockStats) {
					stats := s.blockStats[blockIdx].Fields
					if filter != nil && !filter.MatchesBlock(stats) {
						i += BlockSize
						continue
					}
					if opts.Filter != nil {
						fs := opts.Filter
						if !matchesFilterSet(fs, stats) {
							i += BlockSize
							continue
						}
					}
				}
			}

			// Check filter
			if filter != nil && !filter.Matches(uint32(i)) {
				i++
				continue
			}

			if opts.Filter != nil {
				if len(s.metadataOffsets) > 0 {
					startOffset := s.metadataOffsets[i]
					endOffset := s.metadataOffsets[i+1]
					if startOffset < endOffset {
						mdBytes := s.metadataBlob[startOffset:endOffset]
						match, err := opts.Filter.MatchesBinary(mdBytes)
						if err != nil {
							i++
							continue
						}
						if !match {
							i++
							continue
						}
					} else {
						i++
						continue
					}
				}
			}

			var dist float32
			var approx bool

			if s.sq != nil {
				// Use codes
				code := s.codes[i*dim : (i+1)*dim]
				var err error
				if s.Metric() == distance.MetricL2 {
					dist, err = s.sq.L2Distance(q, code)
				} else {
					dist, err = s.sq.DotProduct(q, code)
				}
				if err != nil {
					return err
				}
				approx = true
			} else if s.pq != nil {
				m := s.pq.NumSubvectors()
				code := s.codes[i*m : (i+1)*m]
				var err error
				dist, err = s.pq.AdcDistance(pqTable, code)
				if err != nil {
					return err
				}
				approx = true
			} else {
				// Use vectors
				vec := s.vectors[i*dim : (i+1)*dim]
				if s.Metric() == distance.MetricL2 {
					dist = distance.SquaredL2(q, vec)
				} else {
					dist = distance.Dot(q, vec)
				}
				approx = false
			}

			cand := searcher.InternalCandidate{
				SegmentID: uint32(s.ID()),
				RowID:     uint32(i),
				Score:     dist,
				Approx:    approx,
			}

			if h.Len() < k {
				h.Push(cand)
			} else {
				top := h.Candidates[0]
				if searcher.InternalCandidateBetter(cand, top, descending) {
					h.ReplaceTop(cand)
				}
			}
			i++
		}
		return nil
	}

	if s.numPartitions > 1 {
		nprobes := opts.NProbes
		if nprobes <= 0 {
			nprobes = 1
		}

		partitions, err := kmeans.FindClosestCentroids(q, s.centroids, dim, nprobes, s.Metric())
		if err != nil {
			return err
		}

		for _, p := range partitions {
			start := int(s.partitionOffsets[p])
			end := int(s.partitionOffsets[p+1])
			if err := scan(start, end); err != nil {
				return err
			}
		}
	} else {
		if err := scan(0, int(s.header.RowCount)); err != nil {
			return err
		}
	}

	return nil
}

func (s *Segment) Rerank(ctx context.Context, q []float32, cands []model.Candidate, dst []model.Candidate) ([]model.Candidate, error) {
	dim := int(s.header.Dim)
	isL2 := s.Metric() == distance.MetricL2

	for _, c := range cands {
		if c.Loc.SegmentID != s.ID() {
			continue
		}
		rowID := int(c.Loc.RowID)
		if rowID >= int(s.header.RowCount) {
			continue
		}

		vec := s.vectors[rowID*dim : (rowID+1)*dim]
		var dist float32
		if isL2 {
			dist = distance.SquaredL2(q, vec)
		} else {
			dist = distance.Dot(q, vec)
		}

		c.Score = dist
		c.Approx = false
		dst = append(dst, c)
	}

	return dst, nil
}

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

	dim := int(s.header.Dim)
	var vectorsBacking []float32
	if fetchVectors {
		vectorsBacking = make([]float32, len(rows)*dim)
	}

	for i, rowID := range rows {
		if rowID >= s.header.RowCount {
			return nil, fmt.Errorf("rowID %d out of bounds", rowID)
		}

		// Fetch ID
		batch.IDs[i] = s.ids[rowID]

		// Fetch Vector
		if fetchVectors {
			vec := s.vectors[int(rowID)*dim : (int(rowID)+1)*dim]
			// Use backing array
			v := vectorsBacking[i*dim : (i+1)*dim]
			copy(v, vec)
			batch.Vectors[i] = v
		}

		// Fetch Metadata
		if fetchMetadata && len(s.metadataOffsets) > 0 && rowID < uint32(len(s.metadataOffsets)-1) {
			start := s.metadataOffsets[rowID]
			end := s.metadataOffsets[rowID+1]
			size := end - start
			if size > 0 {
				mdBytes := s.metadataBlob[start:end]
				var md metadata.Document
				if err := md.UnmarshalBinary(mdBytes); err != nil {
					return nil, fmt.Errorf("failed to unmarshal metadata for row %d: %w", rowID, err)
				}
				batch.Metadatas[i] = md
			}
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
		if rowID >= s.header.RowCount {
			return fmt.Errorf("rowID %d out of bounds", rowID)
		}
		dst[i] = s.ids[rowID]
	}
	return nil
}

// Iterate iterates over all vectors in the segment.
// The context is used for cancellation during long iterations.
func (s *Segment) Iterate(ctx context.Context, fn func(rowID uint32, id model.ID, vec []float32, md metadata.Document, payload []byte) error) error {
	dim := int(s.header.Dim)
	for i := 0; i < int(s.header.RowCount); i++ {
		// Periodic context check (every 256 rows)
		if i&255 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		vec := s.vectors[i*dim : (i+1)*dim]
		id := s.ids[i]

		var md metadata.Document
		if len(s.metadataOffsets) > 0 {
			start := s.metadataOffsets[i]
			end := s.metadataOffsets[i+1]
			if start < end {
				mdBytes := s.metadataBlob[start:end]
				if err := md.UnmarshalBinary(mdBytes); err != nil {
					return err
				}
			}
		}

		var payload []byte
		if s.payloadBlob != nil && uint32(i) < s.payloadCount {
			start := s.payloadOffsets[i]
			end := s.payloadOffsets[i+1]
			size := end - start
			dataOffset := 4 + uint64(s.payloadCount+1)*8 + start

			payload = make([]byte, size)
			if _, err := s.payloadBlob.ReadAt(ctx, payload, int64(dataOffset)); err != nil {
				return err
			}
		}

		if err := fn(uint32(i), id, vec, md, payload); err != nil {
			return err
		}
	}
	return nil
}

// EvaluateFilter returns a bitmap of rows matching the filter.
func (s *Segment) EvaluateFilter(ctx context.Context, filter *metadata.FilterSet) (segment.Bitmap, error) {
	if filter == nil || len(filter.Filters) == 0 {
		return nil, nil // All matches
	}

	// Fast path: use inverted index if available
	if s.metadataIndex != nil {
		return s.metadataIndex.EvaluateFilter(filter), nil
	}

	// Check context before expensive scan
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Slow path: scan all documents (fallback when index not built)
	result := imetadata.GetBitmap() // Use pooled bitmap

	// Periodic cancellation check interval
	const checkInterval = 1024
	lastCheck := 0

	for i := 0; i < int(s.header.RowCount); i++ {
		// Periodic context check
		if i-lastCheck >= checkInterval {
			lastCheck = i
			select {
			case <-ctx.Done():
				imetadata.PutBitmap(result)
				return nil, ctx.Err()
			default:
			}
		}

		var md metadata.Document
		if len(s.metadataOffsets) > 0 {
			start := s.metadataOffsets[i]
			end := s.metadataOffsets[i+1]
			if start < end {
				mdBytes := s.metadataBlob[start:end]
				if err := md.UnmarshalBinary(mdBytes); err != nil {
					imetadata.PutBitmap(result)
					return nil, err
				}
			}
		} else {
			continue // No metadata, cannot match (unless checking for missing)
		}

		if filter.Matches(md) {
			result.Add(uint32(i))
		}
	}

	return result, nil
}

// Advise hints the kernel about access patterns.
func (s *Segment) Advise(pattern segment.AccessPattern) error {
	return nil
}

func matchesFilterSet(fs *metadata.FilterSet, stats map[string]segment.FieldStats) bool {
	if fs == nil {
		return true
	}
	for _, f := range fs.Filters {
		s, ok := stats[f.Key]
		if !ok {
			if f.Value.Kind == metadata.KindInt || f.Value.Kind == metadata.KindFloat {
				return false
			}
			continue
		}

		val := f.Value.F64
		if f.Value.Kind == metadata.KindInt {
			val = float64(f.Value.I64)
		}

		switch f.Operator {
		case metadata.OpEqual:
			if val < s.Min || val > s.Max {
				return false
			}
		case metadata.OpGreaterThan:
			if s.Max <= val {
				return false
			}
		case metadata.OpGreaterEqual:
			if s.Max < val {
				return false
			}
		case metadata.OpLessThan:
			if s.Min >= val {
				return false
			}
		case metadata.OpLessEqual:
			if s.Min > val {
				return false
			}
		}
	}
	return true
}
