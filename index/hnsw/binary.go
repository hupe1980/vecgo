package hnsw

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sync"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/hupe1980/vecgo/internal/visited"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

func init() {
	index.RegisterBinaryLoader(persistence.IndexTypeHNSW, func(r io.Reader) (index.Index, error) {
		h := &HNSW{}
		if err := h.ReadFromWithOptions(r, DefaultOptions); err != nil {
			return nil, err
		}
		return h, nil
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

// SaveToFile saves the HNSW index to a file using a high-performance binary format.
func (h *HNSW) SaveToFile(filename string) error {
	return persistence.SaveToFile(filename, func(w io.Writer) error {
		_, err := h.WriteTo(w)
		return err
	})
}

// LoadFromFile loads the HNSW index from a file.
func LoadFromFile(filename string, opts Options) (*HNSW, error) {
	h := &HNSW{}
	err := persistence.LoadFromFile(filename, func(r io.Reader) error {
		return h.ReadFromWithOptions(r, opts)
	})
	if err != nil {
		return nil, err
	}
	return h, nil
}

// WriteTo writes the HNSW index to a writer in binary format.
func (h *HNSW) WriteTo(w io.Writer) (int64, error) {
	h.freeListMu.Lock()
	defer h.freeListMu.Unlock()

	cw := &countingWriter{w: w}
	writer := persistence.NewBinaryIndexWriter(cw)

	// Calculate actual node count (max ID + 1)
	nextID := h.nextIDAtomic.Load()

	// Write file header
	header := &persistence.FileHeader{
		VectorCount: uint32(int64(nextID) - int64(len(h.freeList))),
		Dimension:   uint32(h.dimensionAtomic.Load()),
		IndexType:   persistence.IndexTypeHNSW,
		DataOffset:  64, // After header
	}
	if err := writer.WriteHeader(header); err != nil {
		return cw.n, err
	}

	// Write HNSW metadata
	metadata := &persistence.HNSWMetadata{
		MaxLayers:    uint16(h.maxLevelAtomic.Load() + 1),
		M:            uint16(h.maxConnectionsPerLayer),
		Ml:           float32(h.layerMultiplier),
		EntryPoint:   h.entryPointAtomic.Load(),
		DistanceFunc: uint8(h.opts.DistanceType),
		Flags: func() uint8 {
			if h.opts.NormalizeVectors {
				return 1
			}
			return 0
		}(),
	}
	if err := persistence.WriteHNSWMetadata(cw, metadata); err != nil {
		return cw.n, err
	}

	// Write nextID
	if err := binary.Write(cw, binary.LittleEndian, nextID); err != nil {
		return cw.n, err
	}

	// Write freeList
	freeListLen := uint32(len(h.freeList))
	if err := binary.Write(cw, binary.LittleEndian, freeListLen); err != nil {
		return cw.n, err
	}
	if freeListLen > 0 {
		if err := writer.WriteUint32Slice(h.freeList); err != nil {
			return cw.n, err
		}
	}

	// Write Arena Size
	arenaSize := h.arena.Size()
	if err := binary.Write(cw, binary.LittleEndian, arenaSize); err != nil {
		return cw.n, err
	}

	// Write Arena Data
	if _, err := cw.Write(h.arena.Buffer()[:arenaSize]); err != nil {
		return cw.n, err
	}

	// Padding for 4-byte alignment of Offsets
	padding := (4 - (arenaSize % 4)) % 4
	if padding > 0 {
		if _, err := cw.Write(make([]byte, padding)); err != nil {
			return cw.n, err
		}
	}

	// Write Offsets
	for id := uint32(0); id < nextID; id++ {
		offset := h.getNodeOffset(id)
		if err := binary.Write(cw, binary.LittleEndian, offset); err != nil {
			return cw.n, err
		}
	}

	// Write Vectors
	for id := uint32(0); id < nextID; id++ {
		vec, ok := h.vectors.GetVector(id)
		if !ok {
			binary.Write(cw, binary.LittleEndian, uint32(0))
			continue
		}
		binary.Write(cw, binary.LittleEndian, uint32(len(vec)))
		writer.WriteFloat32Slice(vec)
	}

	return cw.n, nil
}

// ReadFrom reads the HNSW index from a reader in binary format using DefaultOptions.
func (h *HNSW) ReadFrom(r io.Reader) (int64, error) {
	cr := &countingReader{r: r}
	err := h.ReadFromWithOptions(cr, DefaultOptions)
	return cr.n, err
}

// ReadFromWithOptions reads the HNSW index from a reader in binary format.
func (h *HNSW) ReadFromWithOptions(r io.Reader, opts Options) error {
	reader := persistence.NewBinaryIndexReader(r)

	// Read and validate header
	header, err := reader.ReadHeader()
	if err != nil {
		return err
	}

	if header.IndexType != persistence.IndexTypeHNSW {
		return fmt.Errorf("invalid index type: expected HNSW, got %d", header.IndexType)
	}

	// Read HNSW metadata
	metadata, err := persistence.ReadHNSWMetadata(r)
	if err != nil {
		return err
	}

	// Initialize basic fields
	if opts.Dimension != 0 && opts.Dimension != int(header.Dimension) {
		return &index.ErrDimensionMismatch{Expected: opts.Dimension, Actual: int(header.Dimension)}
	}
	h.opts = opts
	dt := index.DistanceType(metadata.DistanceFunc)
	if dt != index.DistanceTypeSquaredL2 && dt != index.DistanceTypeCosine && dt != index.DistanceTypeDotProduct {
		return fmt.Errorf("unsupported distance type in HNSW index: %d", metadata.DistanceFunc)
	}
	h.opts.DistanceType = dt
	h.opts.NormalizeVectors = (metadata.Flags & 1) != 0
	if h.opts.DistanceType == index.DistanceTypeCosine {
		h.opts.NormalizeVectors = true
	}
	h.opts.M = int(metadata.M)
	h.opts.Dimension = int(header.Dimension)
	h.maxConnectionsPerLayer = int(metadata.M)
	h.maxConnectionsLayer0 = 2 * int(metadata.M)
	h.layerMultiplier = float64(metadata.Ml)
	h.dimensionAtomic.Store(int32(header.Dimension))
	h.entryPointAtomic.Store(metadata.EntryPoint)
	h.maxLevelAtomic.Store(int32(metadata.MaxLayers) - 1)
	if opts.Vectors != nil {
		h.vectors = opts.Vectors
	} else {
		h.vectors = columnar.New(int(header.Dimension))
	}

	// Read nextID
	nextIDSlice, err := reader.ReadUint32Slice(1)
	if err != nil {
		return err
	}
	h.nextIDAtomic.Store(nextIDSlice[0])

	// Read freeList
	freeListLenSlice, err := reader.ReadUint32Slice(1)
	if err != nil {
		return err
	}
	freeListLen := freeListLenSlice[0]
	if freeListLen > 0 {
		h.freeList, err = reader.ReadUint32Slice(int(freeListLen))
		if err != nil {
			return err
		}
	} else {
		h.freeList = []uint32{}
	}

	// Set countAtomic
	h.countAtomic.Store(int64(h.nextIDAtomic.Load()) - int64(len(h.freeList)))

	// Read Arena Size
	var arenaSize uint32
	if err := binary.Read(r, binary.LittleEndian, &arenaSize); err != nil {
		return err
	}

	// Initialize Arena
	h.arena = arena.NewFlat(int(arenaSize))
	// Read Arena Data
	if _, err := io.ReadFull(r, h.arena.Buffer()[:arenaSize]); err != nil {
		return err
	}
	h.arena.SetSize(arenaSize)

	// Skip padding
	padding := (4 - (arenaSize % 4)) % 4
	if padding > 0 {
		if _, err := io.ReadFull(r, make([]byte, padding)); err != nil {
			return err
		}
	}

	// Read Offsets
	nextID := h.nextIDAtomic.Load()
	for id := uint32(0); id < nextID; id++ {
		var offset uint32
		if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
			return err
		}
		if offset != 0 {
			h.setNodeOffset(id, offset)
		}
	}

	// Read Vectors
	for id := uint32(0); id < nextID; id++ {
		var vecLen uint32
		if err := binary.Read(r, binary.LittleEndian, &vecLen); err != nil {
			return err
		}
		if vecLen > 0 {
			vec := make([]float32, vecLen)
			if err := binary.Read(r, binary.LittleEndian, vec); err != nil {
				return err
			}
			h.vectors.SetVector(id, vec)
		}
	}

	// Initialize layout
	h.layout = newNodeLayout(h.opts.M)
	h.shardedLocks = make([]sync.RWMutex, 1024)

	// Initialize distance function
	h.distanceFunc = index.NewDistanceFunc(h.opts.DistanceType)

	// Initialize pools
	h.minQueuePool = &sync.Pool{
		New: func() any {
			return queue.NewMin(h.opts.EF)
		},
	}
	h.maxQueuePool = &sync.Pool{
		New: func() any {
			return queue.NewMax(h.opts.EF)
		},
	}
	h.visitedPool = &sync.Pool{
		New: func() any {
			return visited.New(1024)
		},
	}

	return nil
}

// MarshalBinary implements encoding.BinaryMarshaler for compatibility.
func (h *HNSW) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := h.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler for compatibility.
func (h *HNSW) UnmarshalBinary(data []byte) error {
	// Use default options - caller should use ReadFromWithOptions for custom options
	_, err := h.ReadFrom(bytes.NewReader(data))
	return err
}
