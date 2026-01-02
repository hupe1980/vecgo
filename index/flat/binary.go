package flat

import (
	"context"
	"fmt"
	"io"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

func init() {
	index.RegisterBinaryLoader(persistence.IndexTypeFlat, func(r io.Reader) (index.Index, error) {
		f := &Flat{}
		if err := f.ReadFromWithOptions(r, DefaultOptions); err != nil {
			return nil, err
		}
		return f, nil
	})
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

type countingReader struct {
	r io.Reader
	n int64
}

func (cr *countingReader) Read(p []byte) (int, error) {
	n, err := cr.r.Read(p)
	cr.n += int64(n)
	return n, err
}

// SaveToFile saves the Flat index to a file using high-performance binary format.
func (f *Flat) SaveToFile(filename string) error {
	return persistence.SaveToFile(filename, func(w io.Writer) error {
		_, err := f.WriteTo(w)
		return err
	})
}

// LoadFromFile loads the Flat index from a file.
func LoadFromFile(filename string, opts Options) (*Flat, error) {
	f := &Flat{}
	err := persistence.LoadFromFile(filename, func(r io.Reader) error {
		return f.ReadFromWithOptions(r, opts)
	})
	if err != nil {
		return nil, err
	}
	return f, nil
}

// WriteTo writes the Flat index to a writer in binary format.
//
// It matches the io.WriterTo interface for toolchain friendliness.
func (f *Flat) WriteTo(w io.Writer) (int64, error) {
	f.writeMu.Lock()
	defer f.writeMu.Unlock()

	maxID := f.maxID.Load()

	cw := &countingWriter{w: w}
	writer := persistence.NewBinaryIndexWriter(cw)

	if err := f.writeHeader(writer, maxID); err != nil {
		return cw.n, err
	}

	// Write freeList length and data (Deprecated: Always 0)
	if err := writer.WriteUint64Slice([]uint64{0}); err != nil {
		return cw.n, err
	}

	// Write node count (maxID)
	if err := writer.WriteUint64Slice([]uint64{uint64(maxID)}); err != nil {
		return cw.n, err
	}

	if err := f.writeMarkers(cw, maxID); err != nil {
		return cw.n, err
	}

	if err := f.writeVectors(writer, maxID); err != nil {
		return cw.n, err
	}

	return cw.n, nil
}

func (f *Flat) writeHeader(writer *persistence.BinaryIndexWriter, maxID uint32) error {
	// Write file header
	// Count active nodes (total - deleted)
	activeCount := 0
	for i := uint32(0); i < maxID; i++ {
		if !f.deleted.Test(i) {
			activeCount++
		}
	}
	activeCountU64, err := conv.IntToUint64(activeCount)
	if err != nil {
		return err
	}
	dimU32, err := conv.IntToUint32(int(f.dimension.Load()))
	if err != nil {
		return err
	}
	header := &persistence.FileHeader{
		VectorCount: activeCountU64,
		Dimension:   dimU32,
		IndexType:   persistence.IndexTypeFlat,
		DataOffset:  64, // After header
	}
	if err := writer.WriteHeader(header); err != nil {
		return err
	}

	// Write Flat metadata: distance type + flags.
	distTypeU32, err := conv.IntToUint32(int(f.opts.DistanceType))
	if err != nil {
		return err
	}
	var flags uint32
	if f.opts.NormalizeVectors {
		flags |= 1
	}
	if err := writer.WriteUint32Slice([]uint32{distTypeU32, flags}); err != nil {
		return err
	}
	return nil
}

func (f *Flat) writeMarkers(w io.Writer, maxID uint32) error {
	// Write Markers (Validity Bitmap)
	// For simplicity and alignment, we use 1 byte per node for now.
	// 0 = nil (deleted), 1 = valid.
	maxIDInt, err := conv.Uint32ToInt(maxID)
	if err != nil {
		return err
	}
	markers := make([]byte, maxIDInt)
	for i := uint32(0); i < maxID; i++ {
		if !f.deleted.Test(i) {
			markers[i] = 1
		}
	}
	if _, err := w.Write(markers); err != nil {
		return err
	}

	// Padding to 4-byte alignment
	padding := (4 - (maxID % 4)) % 4
	if padding > 0 {
		if _, err := w.Write(make([]byte, padding)); err != nil {
			return err
		}
	}
	return nil
}

func (f *Flat) writeVectors(writer *persistence.BinaryIndexWriter, maxID uint32) error {
	// Write Vectors (Contiguous)
	zeroVec := make([]float32, f.opts.Dimension)
	for i := uint32(0); i < maxID; i++ {
		if f.deleted.Test(i) {
			if err := writer.WriteFloat32Slice(zeroVec); err != nil {
				return err
			}
			continue
		}
		vec, err := f.VectorByID(context.Background(), core.LocalID(i))
		if err != nil {
			return err
		}
		if err := writer.WriteFloat32Slice(vec); err != nil {
			return err
		}
	}
	return nil
}

// ReadFrom reads the Flat index from a reader in binary format using DefaultOptions.
//
// It matches the io.ReaderFrom interface for toolchain friendliness.
func (f *Flat) ReadFrom(r io.Reader) (int64, error) {
	cr := &countingReader{r: r}
	err := f.ReadFromWithOptions(cr, DefaultOptions)
	return cr.n, err
}

// ReadFromWithOptions reads the Flat index from a reader in binary format.
func (f *Flat) ReadFromWithOptions(r io.Reader, opts Options) error {
	reader := persistence.NewBinaryIndexReader(r)

	header, err := f.readAndValidateHeader(reader, opts)
	if err != nil {
		return err
	}

	if err := f.readMetadata(reader); err != nil {
		return err
	}
	f.vectors = columnar.New(int(header.Dimension))

	// Read freeList (Deprecated: Ignore)
	if err := f.skipFreeList(reader); err != nil {
		return err
	}

	nodeCount, err := f.readNodeCount(reader)
	if err != nil {
		return err
	}

	markers, err := f.readMarkers(r, nodeCount)
	if err != nil {
		return err
	}

	// Initialize state
	nodeCountU32, err := conv.Uint64ToUint32(nodeCount)
	if err != nil {
		return err
	}
	f.maxID.Store(nodeCountU32)

	return f.readVectors(reader, nodeCount, markers, header.Dimension)
}

func (f *Flat) readAndValidateHeader(reader *persistence.BinaryIndexReader, opts Options) (*persistence.FileHeader, error) {
	header, err := reader.ReadHeader()
	if err != nil {
		return nil, err
	}

	if header.IndexType != persistence.IndexTypeFlat {
		return nil, fmt.Errorf("invalid index type: expected Flat, got %d", header.IndexType)
	}

	// Initialize opts (caller may supply non-persisted settings).
	// Dimension and DistanceType are persisted and treated as authoritative.
	if opts.Dimension != 0 && opts.Dimension != int(header.Dimension) {
		return nil, &index.ErrDimensionMismatch{Expected: opts.Dimension, Actual: int(header.Dimension)}
	}
	f.opts = opts
	dimInt, err := conv.Uint32ToInt(header.Dimension)
	if err != nil {
		return nil, err
	}
	f.opts.Dimension = dimInt
	dimI32, err := conv.Uint32ToInt32(header.Dimension)
	if err != nil {
		return nil, err
	}
	f.dimension.Store(dimI32)

	if f.deleted == nil {
		f.deleted = bitset.New(1024)
	}
	return header, nil
}

func (f *Flat) readMetadata(reader *persistence.BinaryIndexReader) error {
	// Read metadata: distance type + flags
	meta, err := reader.ReadUint32Slice(2)
	if err != nil {
		return err
	}
	dt := index.DistanceType(meta[0])
	flags := meta[1]
	if dt != index.DistanceTypeSquaredL2 && dt != index.DistanceTypeCosine && dt != index.DistanceTypeDotProduct {
		return &index.ErrInvalidDistanceType{DistanceType: dt}
	}
	f.opts.DistanceType = dt
	f.opts.NormalizeVectors = (flags & 1) != 0
	if f.opts.DistanceType == index.DistanceTypeCosine {
		// Enforce cosine normalization for correctness.
		f.opts.NormalizeVectors = true
	}
	f.distanceFunc = index.NewDistanceFunc(f.opts.DistanceType)
	return nil
}

func (f *Flat) skipFreeList(reader *persistence.BinaryIndexReader) error {
	// Read freeList (Deprecated: Ignore)
	freeListLenSlice, err := reader.ReadUint64Slice(1)
	if err != nil {
		return err
	}
	freeListLen := freeListLenSlice[0]
	if freeListLen > 100_000_000 {
		return fmt.Errorf("freeList length %d exceeds limit", freeListLen)
	}

	if freeListLen > 0 {
		// Consume and discard free list data
		freeListLenInt, err := conv.Uint64ToInt(freeListLen)
		if err != nil {
			return err
		}
		if _, err := reader.ReadUint64Slice(freeListLenInt); err != nil {
			return err
		}
	}
	return nil
}

func (f *Flat) readNodeCount(reader *persistence.BinaryIndexReader) (uint64, error) {
	// Read node count
	nodeCountSlice, err := reader.ReadUint64Slice(1)
	if err != nil {
		return 0, err
	}
	nodeCount := nodeCountSlice[0]
	if nodeCount > 100_000_000 {
		return 0, fmt.Errorf("node count %d exceeds limit", nodeCount)
	}
	return nodeCount, nil
}

func (f *Flat) readMarkers(r io.Reader, nodeCount uint64) ([]byte, error) {
	// Read Markers
	nodeCountInt, err := conv.Uint64ToInt(nodeCount)
	if err != nil {
		return nil, err
	}
	markers := make([]byte, nodeCountInt)
	if _, err := io.ReadFull(r, markers); err != nil {
		return nil, fmt.Errorf("failed to read markers: %w", err)
	}

	// Skip padding
	padding := (4 - (nodeCount % 4)) % 4
	if padding > 0 {
		paddingInt, err := conv.Uint64ToInt(padding)
		if err != nil {
			return nil, err
		}
		if _, err := io.ReadFull(r, make([]byte, paddingInt)); err != nil {
			return nil, fmt.Errorf("failed to read padding: %w", err)
		}
	}
	return markers, nil
}

func (f *Flat) readVectors(reader *persistence.BinaryIndexReader, nodeCount uint64, markers []byte, dimension uint32) error {
	// Read Vectors
	vecSize, err := conv.Uint32ToInt(dimension)
	if err != nil {
		return err
	}
	vec := make([]float32, vecSize)

	for i := uint64(0); i < nodeCount; i++ {
		// Read vector
		if err := reader.ReadFloat32SliceInto(vec); err != nil {
			return fmt.Errorf("failed to read node vector: %w", err)
		}

		iU32, err := conv.Uint64ToUint32(i)
		if err != nil {
			return err
		}

		if markers[i] == 0 {
			f.deleted.Set(iU32)
			continue
		}

		// Copy vector
		vecCopy := make([]float32, vecSize)
		copy(vecCopy, vec)

		if err := f.vectors.SetVector(core.LocalID(iU32), vecCopy); err != nil {
			return fmt.Errorf("failed to store node vector: %w", err)
		}
	}

	return nil
}
