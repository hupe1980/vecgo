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
	"github.com/hupe1980/vecgo/internal/arena"
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

	// freeList
	freeListLen, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}
	freeList, err := r.ReadUint64SliceCopy(int(freeListLen))
	if err != nil {
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
	h.entryPointAtomic.Store(entryPoint)
	h.maxLevelAtomic.Store(int32(maxLayers) - 1)
	h.nextIDAtomic.Store(nextID)
	h.freeList = freeList
	h.vectors = zerocopy.New(int(hdr.Dimension))
	h.tombstones = ts

	// Initialize runtime fields
	h.shardedLocks = make([]sync.RWMutex, 1024)
	h.minQueuePool = &sync.Pool{
		New: func() any { return queue.NewMin(h.opts.EF) },
	}
	h.maxQueuePool = &sync.Pool{
		New: func() any { return queue.NewMax(h.opts.EF) },
	}
	h.visitedPool = &sync.Pool{
		New: func() any { return visited.New(1024) },
	}
	h.layout = newNodeLayout(h.opts.M)
	h.distanceFunc = index.NewDistanceFunc(dt)
	h.rng = rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set countAtomic
	h.countAtomic.Store(int64(nextID) - int64(len(freeList)) - int64(ts.Count()))

	// Read Arena Size
	arenaSize, err := r.ReadUint64()
	if err != nil {
		return nil, 0, err
	}

	// Initialize Arena (Zero-Copy)
	arenaData, err := r.ReadBytes(int(arenaSize))
	if err != nil {
		return nil, 0, err
	}
	h.arena = arena.NewFlatFromBytes(arenaData)
	h.arena.SetSize(arenaSize)

	// Skip padding
	padding := (8 - (arenaSize % 8)) % 8
	if padding > 0 {
		if _, err := r.ReadBytes(int(padding)); err != nil {
			return nil, 0, err
		}
	}

	// Read Offsets (Zero-Copy)
	offsetsData, err := r.ReadBytes(int(nextID) * 8)
	if err != nil {
		return nil, 0, err
	}

	if len(offsetsData) > 0 {
		h.mmapOffsets = unsafe.Slice((*uint64)(unsafe.Pointer(&offsetsData[0])), int(nextID))
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

	// Initialize layout
	h.layout = newNodeLayout(h.opts.M)
	h.shardedLocks = make([]sync.RWMutex, 1024)

	h.distanceFunc = index.NewDistanceFunc(h.opts.DistanceType)
	h.minQueuePool = &sync.Pool{New: func() any { return queue.NewMin(h.opts.EF) }}
	h.maxQueuePool = &sync.Pool{New: func() any { return queue.NewMax(h.opts.EF) }}
	h.visitedPool = &sync.Pool{New: func() any { return visited.New(1024) }}

	return h, r.Offset(), nil
}
