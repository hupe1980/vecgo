package flat

import (
	"fmt"
	"unsafe"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
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
	f.deleted = bitset.New(1024)

	// freeList (Deprecated: Ignore)
	freeListLen, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}
	// Consume and discard free list data
	if _, err := r.ReadUint64SliceCopy(int(freeListLen)); err != nil {
		return nil, 0, err
	}

	nodeCount, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}
	f.maxID.Store(nodeCount)

	// Read Markers
	markers, err := r.ReadBytes(int(nodeCount))
	if err != nil {
		return nil, 0, err
	}

	// Skip padding
	padding := (4 - (nodeCount % 4)) % 4
	if padding > 0 {
		if _, err := r.ReadBytes(int(padding)); err != nil {
			return nil, 0, err
		}
	}

	// Read Vectors (Zero-Copy)
	vecSize := int(h.Dimension) * 4
	totalVecBytes := int(nodeCount) * vecSize
	vecData, err := r.ReadBytes(totalVecBytes)
	if err != nil {
		return nil, 0, err
	}

	if len(vecData) > 0 {
		vecs := unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), int(nodeCount)*int(h.Dimension))
		if zs, ok := f.vectors.(*zerocopy.Store); ok {
			zs.SetData(vecs)
		} else {
			return nil, 0, fmt.Errorf("internal error: expected zerocopy.Store")
		}
	}

	// Reconstruct deleted
	for i := 0; i < int(nodeCount); i++ {
		if markers[i] == 0 {
			f.deleted.Set(uint64(i))
		}
	}

	return f, r.Offset(), nil
}
