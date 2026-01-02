package hnsw

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"time"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/conv"
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
	h.initPools()
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
	g := h.currentGraph.Load()

	cw := &countingWriter{w: w}
	writer := persistence.NewBinaryIndexWriter(cw)

	// Calculate actual node count (max ID + 1)
	nextID := g.nextIDAtomic.Load()

	// Write file header
	tombstonesCount := uint64(g.tombstones.Count())

	dimU32, err := conv.Int32ToUint32(h.dimensionAtomic.Load())
	if err != nil {
		return cw.n, err
	}

	header := &persistence.FileHeader{
		VectorCount: uint64(nextID) - tombstonesCount,
		Dimension:   dimU32,
		IndexType:   persistence.IndexTypeHNSW,
		DataOffset:  64, // After header
	}
	if err := writer.WriteHeader(header); err != nil {
		return cw.n, err
	}

	// Write HNSW metadata
	maxLayersU16, err := conv.Int32ToUint16(g.maxLevelAtomic.Load() + 1)
	if err != nil {
		return cw.n, err
	}
	mU16, err := conv.IntToUint16(h.maxConnectionsPerLayer)
	if err != nil {
		return cw.n, err
	}
	distFuncU8, err := conv.IntToUint8(int(h.opts.DistanceType))
	if err != nil {
		return cw.n, err
	}

	metadata := &persistence.HNSWMetadata{
		MaxLayers:    maxLayersU16,
		M:            mU16,
		Ml:           float32(h.layerMultiplier),
		EntryPoint:   uint64(g.entryPointAtomic.Load()),
		DistanceFunc: distFuncU8,
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
	if err := binary.Write(cw, binary.LittleEndian, uint64(nextID)); err != nil {
		return cw.n, err
	}

	// Write freeList (Deprecated: Always 0)
	if err := binary.Write(cw, binary.LittleEndian, uint64(0)); err != nil {
		return cw.n, err
	}

	// Write Tombstones
	if _, err := g.tombstones.WriteTo(cw); err != nil {
		return cw.n, err
	}

	// Write Graph Data
	for id := uint32(0); id < nextID; id++ {
		node := h.getNode(g, core.LocalID(id))
		if node == nil {
			// Deleted or unused ID
			if err := binary.Write(cw, binary.LittleEndian, int32(-1)); err != nil {
				return cw.n, err
			}
			continue
		}

		// Write Level
		level := node.Level(g.arena)
		levelI32, err := conv.IntToInt32(level)
		if err != nil {
			return cw.n, err
		}
		if err := binary.Write(cw, binary.LittleEndian, levelI32); err != nil {
			return cw.n, err
		}

		// Write Connections
		for layer := 0; layer <= level; layer++ {
			conns := h.getConnections(g, core.LocalID(id), layer)
			count, err := conv.IntToUint32(len(conns))
			if err != nil {
				return cw.n, err
			}
			if err := binary.Write(cw, binary.LittleEndian, count); err != nil {
				return cw.n, err
			}
			if count > 0 {
				ids := make([]uint64, len(conns))
				for i, c := range conns {
					ids[i] = uint64(c.ID)
				}
				if err := writer.WriteUint64Slice(ids); err != nil {
					return cw.n, err
				}
			}
		}
	}

	// Write Vectors
	// We write vectors contiguously without length prefix for mmap compatibility.
	// Missing vectors (freed IDs) are written as zeros.
	zeroVec := make([]float32, h.opts.Dimension)
	for id := uint32(0); id < nextID; id++ {
		vec, ok := h.vectors.GetVector(core.LocalID(id))
		if !ok {
			if err := writer.WriteFloat32Slice(zeroVec); err != nil {
				return cw.n, err
			}
			continue
		}
		if err := writer.WriteFloat32Slice(vec); err != nil {
			return cw.n, err
		}
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
	// Wrap r in countingReader to track offset for alignment
	cr := &countingReader{r: r}
	reader := persistence.NewBinaryIndexReader(cr)

	// Read and validate header
	header, err := reader.ReadHeader()
	if err != nil {
		return err
	}

	if header.IndexType != persistence.IndexTypeHNSW {
		return fmt.Errorf("invalid index type: expected HNSW, got %d", header.IndexType)
	}

	// Read HNSW metadata
	metadata, err := persistence.ReadHNSWMetadata(cr)
	if err != nil {
		return err
	}

	// Initialize basic fields
	dimInt, err := conv.Uint32ToInt(header.Dimension)
	if err != nil {
		return err
	}
	if opts.Dimension != 0 && opts.Dimension != dimInt {
		return &index.ErrDimensionMismatch{Expected: opts.Dimension, Actual: dimInt}
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
	h.opts.Dimension = dimInt
	h.maxConnectionsPerLayer = int(metadata.M)
	h.maxConnectionsLayer0 = 2 * int(metadata.M)
	h.layerMultiplier = float64(metadata.Ml)
	dimI32, err := conv.Uint32ToInt32(header.Dimension)
	if err != nil {
		return err
	}
	h.dimensionAtomic.Store(dimI32)

	g, err := newGraph(h.opts.InitialArenaSize)
	if err != nil {
		return err
	}
	h.currentGraph.Store(g)

	epU32, err := conv.Uint64ToUint32(metadata.EntryPoint)
	if err != nil {
		return err
	}
	g.entryPointAtomic.Store(epU32)
	g.maxLevelAtomic.Store(int32(metadata.MaxLayers) - 1)

	// Initialize runtime fields
	h.distanceFunc = index.NewDistanceFunc(h.opts.DistanceType)
	h.growNodes(g, 0)
	h.initPools()

	if h.opts.RandomSeed != nil {
		h.rng = rand.New(rand.NewSource(*h.opts.RandomSeed))
	} else {
		h.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	if opts.Vectors != nil {
		h.vectors = opts.Vectors
	} else {
		h.vectors = columnar.New(dimInt)
	}

	// Read nextID
	nextIDSlice, err := reader.ReadUint64Slice(1)
	if err != nil {
		return err
	}
	nextIDU32, err := conv.Uint64ToUint32(nextIDSlice[0])
	if err != nil {
		return err
	}
	g.nextIDAtomic.Store(nextIDU32)
	g.tombstones.Grow(nextIDU32)

	// Read freeList (Deprecated: Ignore)
	freeListLenSlice, err := reader.ReadUint64Slice(1)
	if err != nil {
		return err
	}
	freeListLen := freeListLenSlice[0]
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

	// Read Tombstones
	if _, err := g.tombstones.ReadFrom(cr); err != nil {
		return err
	}

	// Set countAtomic
	g.countAtomic.Store(int64(g.nextIDAtomic.Load()) - int64(g.tombstones.Count()))

	// Read Graph Data
	nextID := g.nextIDAtomic.Load()
	for id := uint32(0); id < nextID; id++ {
		var level int32
		if err := binary.Read(cr, binary.LittleEndian, &level); err != nil {
			return err
		}

		if level < 0 {
			continue
		}

		node, err := h.newNode(g, int(level))
		if err != nil {
			return err
		}
		h.setNode(g, core.LocalID(id), node)

		for layer := 0; layer <= int(level); layer++ {
			var count uint32
			if err := binary.Read(cr, binary.LittleEndian, &count); err != nil {
				return err
			}
			if count > 0 {
				countInt, err := conv.Uint32ToInt(count)
				if err != nil {
					return err
				}
				ids, err := reader.ReadUint64Slice(countInt)
				if err != nil {
					return err
				}
				conns := make([]Neighbor, len(ids))
				for i, id := range ids {
					idU32, err := conv.Uint64ToUint32(id)
					if err != nil {
						return err
					}
					conns[i] = Neighbor{ID: core.LocalID(idU32)}
				}
				h.setConnections(g, core.LocalID(id), layer, conns)
			}
		}
	}

	// Read Vectors
	// Vectors are stored contiguously (Dimension * 4 bytes each)
	vecSize := h.opts.Dimension

	// We can read all vectors in one go if the reader supports it,
	// but for now let's read one by one to populate the store.
	// Optimization: If h.vectors supports SetData (zerocopy), we could read all at once.
	// But h.vectors is an interface.

	vec := make([]float32, vecSize)
	for id := uint32(0); id < nextID; id++ {
		if err := reader.ReadFloat32SliceInto(vec); err != nil {
			return err
		}
		// We need to copy because vec is reused
		vecCopy := make([]float32, vecSize)
		copy(vecCopy, vec)
		if err := h.vectors.SetVector(core.LocalID(id), vecCopy); err != nil {
			return err
		}
	}

	// Recompute distances for all connections
	for id := uint32(0); id < nextID; id++ {
		node := h.getNode(g, core.LocalID(id))
		if node == nil {
			continue
		}
		vec, ok := h.vectors.GetVector(core.LocalID(id))
		if !ok {
			continue
		}
		for l := 0; l <= node.Level(g.arena); l++ {
			conns := h.getConnections(g, core.LocalID(id), l)
			for i := range conns {
				conns[i].Dist = h.dist(vec, conns[i].ID)
			}
			h.setConnections(g, core.LocalID(id), l, conns)
		}
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
