package flat

import (
	"fmt"
	"unsafe"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/vectorstore/zerocopy"
)

func init() {
	index.RegisterMmapBinaryLoader(persistence.IndexTypeFlat, loadFlatMmap)
}

func loadFlatMmap(data []byte) (index.Index, int, error) {
	r := persistence.NewSliceReader(data)

	f, err := readHeaderAndOptions(r)
	if err != nil {
		return nil, 0, err
	}

	if err := skipFreeList(r); err != nil {
		return nil, 0, err
	}

	nodeCount, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}
	nodeCountU32, err := conv.Uint64ToUint32(nodeCount)
	if err != nil {
		return nil, 0, err
	}
	f.maxID.Store(nodeCountU32)

	markers, err := readMarkers(r, nodeCount)
	if err != nil {
		return nil, 0, err
	}

	if err := readVectors(r, f, nodeCount); err != nil {
		return nil, 0, err
	}

	if err := reconstructDeleted(f, markers); err != nil {
		return nil, 0, err
	}

	return f, r.Offset(), nil
}

func readHeaderAndOptions(r *persistence.SliceReader) (*Flat, error) {
	h, err := r.ReadFileHeader()
	if err != nil {
		return nil, err
	}
	if h.IndexType != persistence.IndexTypeFlat {
		return nil, fmt.Errorf("invalid index type: expected Flat, got %d", h.IndexType)
	}

	f := &Flat{}
	f.opts = DefaultOptions
	dimInt, err := conv.Uint32ToInt(h.Dimension)
	if err != nil {
		return nil, err
	}
	f.opts.Dimension = dimInt
	dimI32, err := conv.Uint32ToInt32(h.Dimension)
	if err != nil {
		return nil, err
	}
	f.dimension.Store(dimI32)

	dtU32, err := r.ReadUint32()
	if err != nil {
		return nil, err
	}
	flags, err := r.ReadUint32()
	if err != nil {
		return nil, err
	}
	dt := index.DistanceType(dtU32)
	if dt != index.DistanceTypeSquaredL2 && dt != index.DistanceTypeCosine && dt != index.DistanceTypeDotProduct {
		return nil, &index.ErrInvalidDistanceType{DistanceType: dt}
	}
	f.opts.DistanceType = dt
	f.opts.NormalizeVectors = (flags & 1) != 0
	if f.opts.DistanceType == index.DistanceTypeCosine {
		f.opts.NormalizeVectors = true
	}
	f.distanceFunc = index.NewDistanceFunc(f.opts.DistanceType)
	f.vectors = zerocopy.New(dimInt)
	f.deleted = bitset.New(1024)
	return f, nil
}

func skipFreeList(r *persistence.SliceReader) error {
	freeListLen, err := r.ReadUint64()
	if err != nil {
		return err
	}
	freeListLenInt, err := conv.Uint64ToInt(freeListLen)
	if err != nil {
		return err
	}
	_, err = r.ReadUint64SliceCopy(freeListLenInt)
	return err
}

func readMarkers(r *persistence.SliceReader, nodeCount uint64) ([]byte, error) {
	nodeCountInt, err := conv.Uint64ToInt(nodeCount)
	if err != nil {
		return nil, err
	}
	markers, err := r.ReadBytes(nodeCountInt)
	if err != nil {
		return nil, err
	}

	padding := (4 - (nodeCount % 4)) % 4
	if padding > 0 {
		paddingInt, err := conv.Uint64ToInt(padding)
		if err != nil {
			return nil, err
		}
		if _, err := r.ReadBytes(paddingInt); err != nil {
			return nil, err
		}
	}
	return markers, nil
}

func readVectors(r *persistence.SliceReader, f *Flat, nodeCount uint64) error {
	nodeCountInt, err := conv.Uint64ToInt(nodeCount)
	if err != nil {
		return err
	}
	vecSize := f.opts.Dimension * 4
	totalVecBytes := nodeCountInt * vecSize
	vecData, err := r.ReadBytes(totalVecBytes)
	if err != nil {
		return err
	}

	if len(vecData) > 0 {
		vecs := unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), nodeCountInt*f.opts.Dimension) //nolint:gosec // unsafe is required for performance
		if zs, ok := f.vectors.(*zerocopy.Store); ok {
			zs.SetData(vecs)
		} else {
			return fmt.Errorf("internal error: expected zerocopy.Store")
		}
	}
	return nil
}

func reconstructDeleted(f *Flat, markers []byte) error {
	for i, m := range markers {
		if m == 0 {
			iU32, err := conv.IntToUint32(i)
			if err != nil {
				return err
			}
			f.deleted.Set(iU32)
		}
	}
	return nil
}
