package flat

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"math"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/internal/kmeans"
	"github.com/hupe1980/vecgo/internal/quantization"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// Writer builds a flat segment.
type Writer struct {
	w         io.Writer
	payloadW  io.Writer
	segmentID model.SegmentID
	dim       int
	metric    distance.Metric
	vectors   []float32
	ids       []model.ID
	metadata  [][]byte // Serialized metadata
	payloads  [][]byte // Raw payload
	k         int      // Number of partitions
	quantType int      // Quantization type (0=None, 1=SQ8, 2=PQ)
	pqM       int      // Number of subvectors for PQ

	// Block stats
	// Computed at Flush time
}

// NewWriter creates a new segment writer.
func NewWriter(w io.Writer, payloadW io.Writer, segID model.SegmentID, dim int, metric distance.Metric, k int, quantType int) *Writer {
	return &Writer{
		w:         w,
		payloadW:  payloadW,
		segmentID: segID,
		dim:       dim,
		metric:    metric,
		k:         k,
		quantType: quantType,
		pqM:       dim / 8, // Default heuristic
	}
}

// SetPQConfig sets the PQ configuration.
func (w *Writer) SetPQConfig(m int) {
	w.pqM = m
}

// Add adds a vector and its ID to the segment.
func (w *Writer) Add(id model.ID, vec []float32, md metadata.Document, payload []byte) error {
	if len(vec) != w.dim {
		return errors.New("dimension mismatch")
	}
	w.vectors = append(w.vectors, vec...)
	w.ids = append(w.ids, id)
	w.payloads = append(w.payloads, payload)

	// Serialize metadata
	if md != nil {
		b, err := md.MarshalBinary()
		if err != nil {
			return err
		}
		w.metadata = append(w.metadata, b)
	} else {
		w.metadata = append(w.metadata, nil)
	}
	return nil
}

// Flush writes the segment to the underlying writer.
// The context can be used for cancellation during kmeans training.
func (w *Writer) Flush(ctx context.Context) error {
	rowCount := uint32(len(w.ids))
	var centroids []float32
	var partitionOffsets []uint32

	// Partitioning logic
	if w.k > 1 && rowCount >= uint32(w.k) {
		// 1. Train Centroids
		// Use all vectors for training for now (can sample later if too slow)
		var err error
		centroids, err = kmeans.TrainKMeans(ctx, w.vectors, w.dim, w.k, w.metric, 10)
		if err != nil {
			return err
		}

		// 2. Assign vectors to partitions
		assignments := make([]int, rowCount)
		partitionCounts := make([]int, w.k)
		for i := 0; i < int(rowCount); i++ {
			vec := w.vectors[i*w.dim : (i+1)*w.dim]
			p, err := kmeans.AssignPartition(vec, centroids, w.dim, w.metric)
			if err != nil {
				return err
			}
			assignments[i] = p
			partitionCounts[p]++
		}

		// 3. Reorder vectors and IDs
		newVectors := make([]float32, len(w.vectors))
		newIDs := make([]model.ID, len(w.ids))
		newMetadata := make([][]byte, len(w.metadata))

		// Calculate start offsets for each partition
		partitionStarts := make([]int, w.k)
		current := 0
		for i := 0; i < w.k; i++ {
			partitionStarts[i] = current
			current += partitionCounts[i]
		}

		// Fill new arrays
		currentOffsets := make([]int, w.k)
		copy(currentOffsets, partitionStarts)

		for i := 0; i < int(rowCount); i++ {
			p := assignments[i]
			idx := currentOffsets[p]

			copy(newVectors[idx*w.dim:(idx+1)*w.dim], w.vectors[i*w.dim:(i+1)*w.dim])
			newIDs[idx] = w.ids[i]
			newMetadata[idx] = w.metadata[i]

			currentOffsets[p]++
		}

		w.vectors = newVectors
		w.ids = newIDs
		w.metadata = newMetadata

		// 4. Prepare partition offsets (start indices)
		// We store k+1 offsets: start of p0, start of p1, ..., end of pk-1
		partitionOffsets = make([]uint32, w.k+1)
		for i := 0; i < w.k; i++ {
			partitionOffsets[i] = uint32(partitionStarts[i])
		}
		partitionOffsets[w.k] = rowCount
	} else {
		// No partitioning
		w.k = 0 // Ensure k is 0 if we didn't partition
	}

	// Quantization logic
	var mins, maxs []float32
	var pqCodebooks []int8
	var pqScales, pqOffsets []float32
	var codes []byte
	quantType := uint8(w.quantType)

	if w.quantType != QuantizationNone && rowCount > 0 {
		// Prepare vectors for training
		vecs := make([][]float32, rowCount)
		for i := 0; i < int(rowCount); i++ {
			vecs[i] = w.vectors[i*w.dim : (i+1)*w.dim]
		}

		if w.quantType == QuantizationSQ8 {
			sq := quantization.NewScalarQuantizer(w.dim)
			if err := sq.Train(vecs); err != nil {
				return err
			}

			mins = sq.Mins()
			maxs = sq.Maxs()

			// Encode
			codes = make([]byte, int(rowCount)*w.dim)
			for i := 0; i < int(rowCount); i++ {
				encoded, err := sq.Encode(vecs[i])
				if err != nil {
					return err
				}
				copy(codes[i*w.dim:], encoded)
			}
		} else if w.quantType == QuantizationPQ {
			pq, err := quantization.NewProductQuantizer(w.dim, w.pqM, 256)
			if err != nil {
				return err
			}
			if err := pq.Train(vecs); err != nil {
				return err
			}
			pqCodebooks, pqScales, pqOffsets = pq.Codebooks()

			// Encode
			codes = make([]byte, int(rowCount)*w.pqM)
			for i := 0; i < int(rowCount); i++ {
				encoded, err := pq.Encode(vecs[i])
				if err != nil {
					return err
				}
				copy(codes[i*w.pqM:], encoded)
			}
		}
	}

	// Calculate Block Stats and Prepare Metadata Blob
	var blockStats []BlockStats
	var metadataOffsets []uint32
	var metadataBlob []byte

	if rowCount > 0 {
		metadataOffsets = make([]uint32, rowCount+1)
		currentOffset := uint32(0)

		// Process in blocks
		for i := 0; i < int(rowCount); i += BlockSize {
			end := i + BlockSize
			if end > int(rowCount) {
				end = int(rowCount)
			}

			// Stats for this block
			stats := BlockStats{
				Fields: make(map[string]segment.FieldStats),
			}

			// Iterate rows in block
			for j := i; j < end; j++ {
				mdBytes := w.metadata[j]
				metadataOffsets[j] = currentOffset
				currentOffset += uint32(len(mdBytes))
				metadataBlob = append(metadataBlob, mdBytes...)

				if len(mdBytes) > 0 {
					var md metadata.Document
					if err := md.UnmarshalBinary(mdBytes); err != nil {
						return err
					}

					// Update stats
					for k, v := range md {
						var val float64
						var isNumeric bool

						if v.Kind == metadata.KindFloat {
							val = v.F64
							isNumeric = true
						} else if v.Kind == metadata.KindInt {
							val = float64(v.I64)
							isNumeric = true
						}

						if isNumeric {
							fs, ok := stats.Fields[k]
							if !ok {
								fs = segment.FieldStats{Min: val, Max: val}
							} else {
								if val < fs.Min {
									fs.Min = val
								}
								if val > fs.Max {
									fs.Max = val
								}
							}
							stats.Fields[k] = fs
						}
					}
				}
			}
			blockStats = append(blockStats, stats)
		}
		metadataOffsets[rowCount] = currentOffset
	}

	blockStatsBytes, err := func() ([]byte, error) {
		var buf []byte
		// Count
		buf = binary.AppendUvarint(buf, uint64(len(blockStats)))
		for _, bs := range blockStats {
			b, err := bs.MarshalBinary()
			if err != nil {
				return nil, err
			}
			buf = binary.AppendUvarint(buf, uint64(len(b)))
			buf = append(buf, b...)
		}
		return buf, nil
	}()
	if err != nil {
		return err
	}

	// Calculate offsets
	centroidSize := uint64(len(centroids)) * 4
	partitionOffsetSize := uint64(len(partitionOffsets)) * 4

	var quantMetaSize uint64
	if w.quantType == QuantizationSQ8 {
		quantMetaSize = uint64(len(mins)+len(maxs)) * 4
	} else if w.quantType == QuantizationPQ {
		// m (4) + k (4) + scales (m*4) + offsets (m*4) + codebooks (len*1)
		quantMetaSize = 4 + 4 + uint64(len(pqScales))*4 + uint64(len(pqOffsets))*4 + uint64(len(pqCodebooks))
	}

	codesSize := uint64(len(codes))

	// Calculate ID Blob
	var pkBlob []byte
	if rowCount > 0 {
		pkBlob = make([]byte, 0, rowCount*8)
		for _, id := range w.ids {
			pkBlob = binary.LittleEndian.AppendUint64(pkBlob, uint64(id))
		}
	}

	centroidOffset := uint64(HeaderSize)
	partitionOffsetOffset := centroidOffset + centroidSize
	quantOffset := partitionOffsetOffset + partitionOffsetSize
	codesOffset := quantOffset + quantMetaSize
	vectorOffset := codesOffset + codesSize
	vectorSize := uint64(len(w.vectors)) * 4
	pkOffset := vectorOffset + vectorSize
	pkSize := uint64(len(pkBlob))
	metadataOffset := pkOffset + pkSize
	metadataSize := uint64(len(metadataOffsets)*4 + len(metadataBlob))
	blockStatsOffset := metadataOffset + metadataSize

	header := &FileHeader{
		Magic:                 MagicNumber,
		Version:               Version,
		SegmentID:             uint64(w.segmentID),
		RowCount:              rowCount,
		Dim:                   uint32(w.dim),
		Metric:                uint8(w.metric),
		NumPartitions:         uint32(w.k),
		QuantizationType:      quantType,
		CentroidOffset:        centroidOffset,
		PartitionOffsetOffset: partitionOffsetOffset,
		QuantizationOffset:    quantOffset,
		CodesOffset:           codesOffset,
		VectorOffset:          vectorOffset,
		PKOffset:              pkOffset,
		MetadataOffset:        metadataOffset,
		BlockStatsOffset:      blockStatsOffset,
	}

	bw := bufio.NewWriter(w.w)

	// 1. Write Header (Placeholder)
	if _, err := bw.Write(header.Encode()); err != nil {
		return err
	}

	// Calculate Checksum of the body
	crc := crc32.New(crc32.MakeTable(crc32.Castagnoli))
	mw := io.MultiWriter(bw, crc)

	// 2. Write Centroids
	for _, v := range centroids {
		if err := binary.Write(mw, binary.LittleEndian, math.Float32bits(v)); err != nil {
			return err
		}
	}

	// 3. Write Partition Offsets
	for _, o := range partitionOffsets {
		if err := binary.Write(mw, binary.LittleEndian, o); err != nil {
			return err
		}
	}

	// 4. Write Quantization Metadata
	if w.quantType == QuantizationSQ8 {
		for _, v := range mins {
			if err := binary.Write(mw, binary.LittleEndian, math.Float32bits(v)); err != nil {
				return err
			}
		}
		for _, v := range maxs {
			if err := binary.Write(mw, binary.LittleEndian, math.Float32bits(v)); err != nil {
				return err
			}
		}
	} else if w.quantType == QuantizationPQ {
		// Write m, k
		if err := binary.Write(mw, binary.LittleEndian, uint32(w.pqM)); err != nil {
			return err
		}
		if err := binary.Write(mw, binary.LittleEndian, uint32(256)); err != nil {
			return err
		}
		// Write scales
		for _, v := range pqScales {
			if err := binary.Write(mw, binary.LittleEndian, math.Float32bits(v)); err != nil {
				return err
			}
		}
		// Write offsets
		for _, v := range pqOffsets {
			if err := binary.Write(mw, binary.LittleEndian, math.Float32bits(v)); err != nil {
				return err
			}
		}
		// Write codebooks
		for _, v := range pqCodebooks {
			if err := binary.Write(mw, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}

	// 5. Write Codes
	if len(codes) > 0 {
		if _, err := mw.Write(codes); err != nil {
			return err
		}
	}

	// 6. Write Vectors
	for _, v := range w.vectors {
		if err := binary.Write(mw, binary.LittleEndian, math.Float32bits(v)); err != nil {
			return err
		}
	}

	// 7. Write IDs
	if len(pkBlob) > 0 {
		if _, err := mw.Write(pkBlob); err != nil {
			return err
		}
	}

	// 8. Write Metadata
	for _, o := range metadataOffsets {
		if err := binary.Write(mw, binary.LittleEndian, o); err != nil {
			return err
		}
	}
	if len(metadataBlob) > 0 {
		if _, err := mw.Write(metadataBlob); err != nil {
			return err
		}
	}

	// 9. Write Block Stats
	if len(blockStatsBytes) > 0 {
		if _, err := mw.Write(blockStatsBytes); err != nil {
			return err
		}
	}

	if err := bw.Flush(); err != nil {
		return err
	}

	// Write Payload
	if w.payloadW != nil && rowCount > 0 {
		// Calculate offsets
		payloadOffsets := make([]uint64, rowCount+1)
		currentOffset := uint64(0)
		for i, p := range w.payloads {
			payloadOffsets[i] = currentOffset
			currentOffset += uint64(len(p))
		}
		payloadOffsets[rowCount] = currentOffset

		// Write Count
		if err := binary.Write(w.payloadW, binary.LittleEndian, rowCount); err != nil {
			return err
		}
		// Write Offsets
		if err := binary.Write(w.payloadW, binary.LittleEndian, payloadOffsets); err != nil {
			return err
		}
		// Write Data
		for _, p := range w.payloads {
			if _, err := w.payloadW.Write(p); err != nil {
				return err
			}
		}
	}

	// Update Header with Checksum if seekable
	if seeker, ok := w.w.(io.WriteSeeker); ok {
		header.Checksum = crc.Sum32()
		if _, err := seeker.Seek(0, io.SeekStart); err != nil {
			return err
		}
		if _, err := seeker.Write(header.Encode()); err != nil {
			return err
		}
		// Seek back to end? Not strictly necessary if we are done, but good practice.
		if _, err := seeker.Seek(0, io.SeekEnd); err != nil {
			return err
		}
	}

	return nil
}
