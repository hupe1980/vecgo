package flat

import (
	"fmt"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/vectorstore/zerocopy"
)

func init() {
	index.RegisterMmapBinaryLoader(persistence.IndexTypeFlat, loadFlatMmap)
}

func loadFlatMmap(data []byte) (index.Index, int, error) {
	r := persistence.NewSliceReader(data)

	h, err := r.ReadFileHeader()
	if err != nil {
		return nil, 0, err
	}
	if h.IndexType != persistence.IndexTypeFlat {
		return nil, 0, fmt.Errorf("invalid index type: expected Flat, got %d", h.IndexType)
	}

	f := &Flat{}
	// Dimension is authoritative.
	f.opts = DefaultOptions
	f.opts.Dimension = int(h.Dimension)
	f.dimension.Store(int32(h.Dimension))

	dtU32, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}
	flags, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}
	dt := index.DistanceType(dtU32)
	if dt != index.DistanceTypeSquaredL2 && dt != index.DistanceTypeCosine && dt != index.DistanceTypeDotProduct {
		return nil, 0, &index.ErrInvalidDistanceType{DistanceType: dt}
	}
	f.opts.DistanceType = dt
	f.opts.NormalizeVectors = (flags & 1) != 0
	if f.opts.DistanceType == index.DistanceTypeCosine {
		f.opts.NormalizeVectors = true
	}
	f.distanceFunc = index.NewDistanceFunc(f.opts.DistanceType)
	f.vectors = zerocopy.New(int(h.Dimension))

	freeListLen, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}
	freeList, err := r.ReadUint32SliceCopy(int(freeListLen))
	if err != nil {
		return nil, 0, err
	}

	nodeCount, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}

	nodes := make([]*Node, int(nodeCount))
	for i := 0; i < int(nodeCount); i++ {
		marker, err := r.ReadUint32()
		if err != nil {
			return nil, 0, fmt.Errorf("failed to read node marker: %w", err)
		}
		if marker == 0 {
			nodes[i] = nil
			continue
		}
		id, err := r.ReadUint32()
		if err != nil {
			return nil, 0, fmt.Errorf("failed to read node id: %w", err)
		}
		vec, err := r.ReadFloat32SliceView(int(h.Dimension))
		if err != nil {
			return nil, 0, fmt.Errorf("failed to read node vector: %w", err)
		}
		if err := f.vectors.SetVector(id, vec); err != nil {
			return nil, 0, fmt.Errorf("failed to store node vector: %w", err)
		}
		nodes[i] = &Node{ID: id}
	}

	f.state.Store(&indexState{nodes: nodes, freeList: freeList})
	return f, r.Offset(), nil
}
