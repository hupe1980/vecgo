package hnsw

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
	"unsafe"

	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/bitset"
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
	entryPoint := binary.LittleEndian.Uint64(metaBytes[8:16])
	dt := index.DistanceType(metaBytes[16])
	flags := metaBytes[17]
	if dt != index.DistanceTypeSquaredL2 && dt != index.DistanceTypeCosine && dt != index.DistanceTypeDotProduct {
		return nil, 0, fmt.Errorf("unsupported distance type in HNSW index: %d", metaBytes[16])
	}

	// nextID
	nextID, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}

	// freeList (Deprecated: Ignore)
	freeListLen, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}
	// Consume and discard free list data
	if _, err := r.ReadUint64SliceCopy(int(freeListLen)); err != nil {
		return nil, 0, err
	}

	// Tombstones
	ts := bitset.New(0)
	remaining := r.Remaining()
	n, err := ts.ReadFrom(bytes.NewReader(remaining))
	if err != nil {
		return nil, 0, err
	}
	r.Advance(int(n))

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

	g := newGraph()
	h.currentGraph.Store(g)

	g.entryPointAtomic.Store(entryPoint)
	g.maxLevelAtomic.Store(int32(maxLayers) - 1)
	g.nextIDAtomic.Store(nextID)
	h.vectors = zerocopy.New(int(hdr.Dimension))
	g.tombstones = ts

	// Initialize runtime fields
	h.initPools()
	h.distanceFunc = index.NewDistanceFunc(dt)
	h.rng = rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set countAtomic
	g.countAtomic.Store(int64(nextID) - int64(ts.Count()))

	// Initialize nodes
	h.growNodes(g, 0)

	// Read Graph Data
	for id := uint64(0); id < nextID; id++ {
		levelBytes, err := r.ReadBytes(4)
		if err != nil {
			return nil, 0, err
		}
		level := int32(binary.LittleEndian.Uint32(levelBytes))

		if level < 0 {
			continue
		}

		node := newNode(int(level))
		h.setNode(g, id, node)

		for layer := 0; layer <= int(level); layer++ {
			countBytes, err := r.ReadBytes(4)
			if err != nil {
				return nil, 0, err
			}
			count := binary.LittleEndian.Uint32(countBytes)

			if count > 0 {
				conns, err := r.ReadUint64SliceCopy(int(count))
				if err != nil {
					return nil, 0, err
				}
				node.setConnections(layer, conns)
			}
		}
	}

	// Read Vectors (Zero-Copy, Contiguous)
	// Format: [vec0][vec1]...[vecN] (no length prefixes)
	vecSize := int(hdr.Dimension) * 4
	totalVecBytes := int(nextID) * vecSize

	vecData, err := r.ReadBytes(totalVecBytes)
	if err != nil {
		return nil, 0, err
	}

	if len(vecData) > 0 {
		// Cast to []float32
		vecs := unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), int(nextID)*int(hdr.Dimension))

		// Set data in zerocopy store
		if zs, ok := h.vectors.(*zerocopy.Store); ok {
			zs.SetData(vecs)
		} else {
			return nil, 0, fmt.Errorf("internal error: expected zerocopy.Store")
		}
	}

	h.distanceFunc = index.NewDistanceFunc(h.opts.DistanceType)
	h.minQueuePool = &sync.Pool{New: func() any { return queue.NewMin(h.opts.EF) }}
	h.maxQueuePool = &sync.Pool{New: func() any { return queue.NewMax(h.opts.EF) }}
	h.visitedPool = &sync.Pool{New: func() any { return visited.New(1024) }}

	return h, r.Offset(), nil
}
