package flat

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/hupe1980/vecgo/index"
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

	st := f.getState()

	cw := &countingWriter{w: w}
	writer := persistence.NewBinaryIndexWriter(cw)

	// Write file header
	header := &persistence.FileHeader{
		VectorCount: uint32(len(st.nodes) - len(st.freeList)),
		Dimension:   uint32(f.dimension.Load()),
		IndexType:   persistence.IndexTypeFlat,
		DataOffset:  64, // After header
	}
	if err := writer.WriteHeader(header); err != nil {
		return cw.n, err
	}

	// Write Flat metadata: distance type + flags.
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(f.opts.DistanceType))
	var flags uint32
	if f.opts.NormalizeVectors {
		flags |= 1
	}
	binary.LittleEndian.PutUint32(buf[4:8], flags)
	if _, err := cw.Write(buf); err != nil {
		return cw.n, err
	}

	// Write freeList length and data
	freeListLen := uint32(len(st.freeList))
	buf4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf4, freeListLen)
	if _, err := cw.Write(buf4); err != nil {
		return cw.n, err
	}
	if freeListLen > 0 {
		if err := writer.WriteUint32Slice(st.freeList); err != nil {
			return cw.n, err
		}
	}

	// Write node count
	nodeCount := uint32(len(st.nodes))
	binary.LittleEndian.PutUint32(buf4, nodeCount)
	if _, err := cw.Write(buf4); err != nil {
		return cw.n, err
	}

	// Write each node
	for _, node := range st.nodes {
		if node == nil {
			// Write nil marker (0)
			binary.LittleEndian.PutUint32(buf4, 0)
			if _, err := cw.Write(buf4); err != nil {
				return cw.n, err
			}
			continue
		}

		// Write non-nil marker (1) and ID (8 bytes total)
		nodeHdr := make([]byte, 8)
		binary.LittleEndian.PutUint32(nodeHdr[0:4], 1) // non-nil marker
		binary.LittleEndian.PutUint32(nodeHdr[4:8], node.ID)
		if _, err := cw.Write(nodeHdr); err != nil {
			return cw.n, err
		}

		// Write vector
		vec, err := f.VectorByID(context.Background(), node.ID)
		if err != nil {
			return cw.n, err
		}
		if err := writer.WriteFloat32Slice(vec); err != nil {
			return cw.n, err
		}
	}

	return cw.n, nil
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

	// Read and validate header
	header, err := reader.ReadHeader()
	if err != nil {
		return err
	}

	if header.IndexType != persistence.IndexTypeFlat {
		return fmt.Errorf("invalid index type: expected Flat, got %d", header.IndexType)
	}

	// Initialize opts (caller may supply non-persisted settings).
	// Dimension and DistanceType are persisted and treated as authoritative.
	if opts.Dimension != 0 && opts.Dimension != int(header.Dimension) {
		return &index.ErrDimensionMismatch{Expected: opts.Dimension, Actual: int(header.Dimension)}
	}
	f.opts = opts
	f.opts.Dimension = int(header.Dimension)
	f.dimension.Store(int32(header.Dimension))

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
	f.vectors = columnar.New(int(header.Dimension))

	// Read freeList
	freeListLenSlice, err := reader.ReadUint32Slice(1)
	if err != nil {
		return err
	}
	freeListLen := freeListLenSlice[0]

	var freeList []uint32
	if freeListLen > 0 {
		freeList, err = reader.ReadUint32Slice(int(freeListLen))
		if err != nil {
			return err
		}
	} else {
		freeList = []uint32{}
	}

	// Read node count
	nodeCountSlice, err := reader.ReadUint32Slice(1)
	if err != nil {
		return err
	}
	nodeCount := nodeCountSlice[0]

	// Pre-allocate nodes array
	nodes := make([]*Node, nodeCount)

	// Read each node
	for i := uint32(0); i < nodeCount; i++ {
		// Read nil marker
		markerSlice, err := reader.ReadUint32Slice(1)
		if err != nil {
			return fmt.Errorf("failed to read node marker: %w", err)
		}
		marker := markerSlice[0]

		if marker == 0 {
			// Nil node (tombstone)
			nodes[i] = nil
			continue
		}

		// Read ID
		idSlice, err := reader.ReadUint32Slice(1)
		if err != nil {
			return fmt.Errorf("failed to read node ID: %w", err)
		}
		nodeID := idSlice[0]

		// Read vector
		vector, err := reader.ReadFloat32Slice(int(header.Dimension))
		if err != nil {
			return fmt.Errorf("failed to read node vector: %w", err)
		}
		if err := f.vectors.SetVector(nodeID, vector); err != nil {
			return fmt.Errorf("failed to store node vector: %w", err)
		}

		nodes[i] = &Node{
			ID: nodeID,
		}
	}

	// Create new state and store it
	newState := &indexState{
		nodes:    nodes,
		freeList: freeList,
	}
	f.state.Store(newState)

	return nil
}
