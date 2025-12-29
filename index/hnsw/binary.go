package hnsw

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"github.com/bits-and-blooms/bitset"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/arena"
	"github.com/hupe1980/vecgo/internal/queue"
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
	h.segmentsMu.RLock()
	h.freeListMu.Lock()
	defer h.segmentsMu.RUnlock()
	defer h.freeListMu.Unlock()

	cw := &countingWriter{w: w}
	writer := persistence.NewBinaryIndexWriter(cw)

	// Calculate actual node count (max ID + 1)
	// We use nextID as the upper bound for iteration
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

	// Pre-allocate buffer for small writes (reduces allocations)
	buf := make([]byte, 4)

	// Write nextID
	binary.LittleEndian.PutUint32(buf, nextID)
	if _, err := cw.Write(buf); err != nil {
		return cw.n, err
	}

	// Write freeList length and data
	freeListLen := uint32(len(h.freeList))
	binary.LittleEndian.PutUint32(buf, freeListLen)
	if _, err := cw.Write(buf); err != nil {
		return cw.n, err
	}
	if freeListLen > 0 {
		if err := writer.WriteUint32Slice(h.freeList); err != nil {
			return cw.n, err
		}
	}

	// Write node count (capacity)
	// The reader expects to read this many nodes.
	// We write nextID as the count, so we iterate 0..nextID-1
	binary.LittleEndian.PutUint32(buf, nextID)
	if _, err := cw.Write(buf); err != nil {
		return cw.n, err
	}

	// Write each node
	for id := uint32(0); id < nextID; id++ {
		node := h.getNode(id) // Lock-free read

		if node == nil {
			// Write nil marker (0)
			binary.LittleEndian.PutUint32(buf, 0)
			if _, err := cw.Write(buf); err != nil {
				return cw.n, err
			}
			continue
		}

		vec, ok := h.vectors.GetVector(node.ID)
		if !ok {
			return cw.n, fmt.Errorf("hnsw: missing vector for node id=%d", node.ID)
		}

		// Write non-nil marker (1), ID, and packed layer/veclen in one go (12 bytes)
		nodeHdr := make([]byte, 12)
		binary.LittleEndian.PutUint32(nodeHdr[0:4], 1) // non-nil marker
		binary.LittleEndian.PutUint32(nodeHdr[4:8], node.ID)
		binary.LittleEndian.PutUint32(nodeHdr[8:12], uint32(node.Layer)<<16|uint32(len(vec)))
		if _, err := cw.Write(nodeHdr); err != nil {
			return cw.n, err
		}

		// Write vector
		if err := writer.WriteFloat32Slice(vec); err != nil {
			return cw.n, err
		}

		// Write connections directly (avoid allocation)
		for i := 0; i <= node.Layer; i++ {
			conns := node.Connections[i].Load()
			if conns == nil {
				binary.LittleEndian.PutUint32(buf, 0)
				if _, err := cw.Write(buf); err != nil {
					return cw.n, err
				}
			} else {
				binary.LittleEndian.PutUint32(buf, uint32(len(*conns)))
				if _, err := cw.Write(buf); err != nil {
					return cw.n, err
				}
				if len(*conns) > 0 {
					if err := writer.WriteUint32Slice(*conns); err != nil {
						return cw.n, err
					}
				}
			}
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

	// Read node count
	nodeCountSlice, err := reader.ReadUint32Slice(1)
	if err != nil {
		return err
	}
	nodeCount := nodeCountSlice[0]

	// Initialize segments
	h.segments = make([]atomic.Pointer[[]atomic.Pointer[Node]], 0)
	h.arena = arena.New(arena.DefaultChunkSize)

	// Read nodes
	for i := uint32(0); i < nodeCount; i++ {
		// Read nil marker
		marker, err := reader.ReadUint32Slice(1)
		if err != nil {
			return err
		}

		if marker[0] == 0 {
			h.growSegments(i)
			continue
		}

		// Read node header
		headerData, err := reader.ReadUint32Slice(2)
		if err != nil {
			return err
		}

		nodeID := headerData[0]
		layer := int(headerData[1] >> 16)
		vecLen := int(headerData[1] & 0xFFFF)

		// Read vector
		vector, err := reader.ReadFloat32Slice(vecLen)
		if err != nil {
			return err
		}

		// Store vector
		if err := h.vectors.SetVector(nodeID, vector); err != nil {
			return err
		}

		// Create node (topology-only)
		node := &Node{
			ID:          nodeID,
			Layer:       layer,
			Connections: make([]atomic.Pointer[[]uint32], layer+1),
		}

		// Read connections
		connections, err := persistence.ReadConnections(r, layer+1)
		if err != nil {
			return err
		}

		// Initialize atomic pointers
		for j := 0; j <= layer; j++ {
			connsCopy := make([]uint32, len(connections[j]))
			copy(connsCopy, connections[j])
			node.Connections[j].Store(&connsCopy)
		}

		h.setNode(i, node)
	}

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
	h.bitsetPool = &sync.Pool{
		New: func() any {
			return &bitset.BitSet{}
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
