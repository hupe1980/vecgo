package hnsw

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/queue"
	"github.com/hupe1980/vecgo/internal/visited"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/vectorstore/zerocopy"
)

func init() {
	index.RegisterMmapBinaryLoader(persistence.IndexTypeHNSW, loadHNSWMmap)
}

func loadHNSWMmap(data []byte) (index.Index, int, error) {
	r := persistence.NewSliceReader(data)

	hdr, err := r.ReadFileHeader()
	if err != nil {
		return nil, 0, err
	}
	if hdr.IndexType != persistence.IndexTypeHNSW {
		return nil, 0, fmt.Errorf("invalid index type: expected HNSW, got %d", hdr.IndexType)
	}

	// Read HNSW metadata (28 bytes - see persistence.HNSWMetadata)
	metaBytes, err := r.ReadBytes(28)
	if err != nil {
		return nil, 0, err
	}
	maxLayers := int(binary.LittleEndian.Uint16(metaBytes[0:2]))
	m := int(binary.LittleEndian.Uint16(metaBytes[2:4]))
	ml := math.Float32frombits(binary.LittleEndian.Uint32(metaBytes[4:8]))
	entryPoint := binary.LittleEndian.Uint32(metaBytes[8:12])
	dt := index.DistanceType(metaBytes[12])
	flags := metaBytes[13]
	if dt != index.DistanceTypeSquaredL2 && dt != index.DistanceTypeCosine && dt != index.DistanceTypeDotProduct {
		return nil, 0, fmt.Errorf("unsupported distance type in HNSW index: %d", metaBytes[12])
	}

	// nextID
	nextID, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}

	// freeList
	freeListLen, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}
	freeList, err := r.ReadUint32SliceCopy(int(freeListLen))
	if err != nil {
		return nil, 0, err
	}

	// node count
	nodeCount, err := r.ReadUint32()
	if err != nil {
		return nil, 0, err
	}

	h := &HNSW{}
	h.opts = DefaultOptions
	h.opts.Dimension = int(hdr.Dimension)
	h.opts.DistanceType = dt
	h.opts.NormalizeVectors = (flags & 1) != 0
	if h.opts.DistanceType == index.DistanceTypeCosine {
		h.opts.NormalizeVectors = true
	}
	h.opts.M = m
	h.maxConnectionsPerLayer = m
	h.maxConnectionsLayer0 = 2 * m
	h.layerMultiplier = float64(ml)
	h.dimensionAtomic.Store(int32(hdr.Dimension))
	h.entryPointAtomic.Store(entryPoint)
	h.maxLevelAtomic.Store(int32(maxLayers) - 1)
	h.nextIDAtomic.Store(nextID)
	h.freeList = freeList
	h.vectors = zerocopy.New(int(hdr.Dimension))

	// Initialize segments
	// h.segments is atomic.Pointer, zero value is nil, which is correct.

	for i := 0; i < int(nodeCount); i++ {
		marker, err := r.ReadUint32()
		if err != nil {
			return nil, 0, err
		}
		if marker == 0 {
			h.growSegments(uint32(i))
			continue
		}

		nodeID, err := r.ReadUint32()
		if err != nil {
			return nil, 0, err
		}
		packed, err := r.ReadUint32()
		if err != nil {
			return nil, 0, err
		}
		layer := int(packed >> 16)
		vecLen := int(packed & 0xFFFF)

		vec, err := r.ReadFloat32SliceView(vecLen)
		if err != nil {
			return nil, 0, err
		}
		if err := h.vectors.SetVector(nodeID, vec); err != nil {
			return nil, 0, err
		}

		node := &Node{
			ID:          nodeID,
			Layer:       layer,
			Connections: make([]atomic.Pointer[[]uint32], layer+1),
		}

		// Read connections (copy ids; connection slices are mutable during updates).
		for j := 0; j <= layer; j++ {
			cnt, err := r.ReadUint32()
			if err != nil {
				return nil, 0, err
			}
			if cnt == 0 {
				var empty []uint32
				node.Connections[j].Store(&empty)
				continue
			}
			ids, err := r.ReadUint32SliceCopy(int(cnt))
			if err != nil {
				return nil, 0, err
			}
			node.Connections[j].Store(&ids)
		}

		h.setNode(uint32(i), node)
	}

	h.distanceFunc = index.NewDistanceFunc(h.opts.DistanceType)
	h.minQueuePool = &sync.Pool{New: func() any { return queue.NewMin(h.opts.EF) }}
	h.maxQueuePool = &sync.Pool{New: func() any { return queue.NewMax(h.opts.EF) }}
	h.visitedPool = &sync.Pool{New: func() any { return visited.New(1024) }}

	return h, r.Offset(), nil
}
