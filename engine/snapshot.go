package engine

import (
	"cmp"
	"slices"
	"sync/atomic"

	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/segment/memtable"
	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/pk"
)

// RefCountedSegment wraps a Segment with a reference count.
type RefCountedSegment struct {
	segment.Segment
	refs    int64
	onClose atomic.Value // stores func()
}

func NewRefCountedSegment(seg segment.Segment) *RefCountedSegment {
	r := &RefCountedSegment{
		Segment: seg,
		refs:    1, // Initial ref
	}
	var f func()
	r.onClose.Store(f)
	return r
}

func (r *RefCountedSegment) IncRef() {
	atomic.AddInt64(&r.refs, 1)
}

func (r *RefCountedSegment) DecRef() {
	if atomic.AddInt64(&r.refs, -1) == 0 {
		_ = r.Segment.Close()
		f := r.onClose.Load().(func())
		if f != nil {
			f()
		}
	}
}

// SetOnClose sets a callback function to be executed when the segment is closed.
// This is typically used to delete the underlying file.
func (r *RefCountedSegment) SetOnClose(f func()) {
	r.onClose.Store(f)
}

// Snapshot represents a consistent view of the database.
type Snapshot struct {
	refs            int64
	segments        map[model.SegmentID]*RefCountedSegment
	sortedSegments  []*RefCountedSegment // Deterministic iteration order
	tombstones      map[model.SegmentID]*imetadata.LocalBitmap
	active          *memtable.MemTable
	activeWatermark uint32 // Limit visibility of active segment rows
	pkIndex         *pk.PersistentIndex
}

func NewSnapshot(active *memtable.MemTable) *Snapshot {
	return &Snapshot{
		refs:            1,
		segments:        make(map[model.SegmentID]*RefCountedSegment),
		sortedSegments:  make([]*RefCountedSegment, 0),
		tombstones:      make(map[model.SegmentID]*imetadata.LocalBitmap),
		active:          active,
		activeWatermark: active.RowCount(),
		pkIndex:         pk.NewPersistentIndex(),
	}
}

func (s *Snapshot) IncRef() {
	atomic.AddInt64(&s.refs, 1)
}

// TryIncRef attempts to increment the reference count.
// Returns true if successful, false if the snapshot is already destroyed (refs == 0).
func (s *Snapshot) TryIncRef() bool {
	for {
		refs := atomic.LoadInt64(&s.refs)
		if refs <= 0 {
			return false
		}
		if atomic.CompareAndSwapInt64(&s.refs, refs, refs+1) {
			return true
		}
	}
}

func (s *Snapshot) DecRef() {
	if atomic.AddInt64(&s.refs, -1) == 0 {
		for _, seg := range s.segments {
			seg.DecRef()
		}
		s.active.DecRef()
	}
}

// RebuildSorted rebuilds the sortedSegments slice.
// Must be called before the snapshot is published or used for search.
func (s *Snapshot) RebuildSorted() {
	s.sortedSegments = make([]*RefCountedSegment, 0, len(s.segments))
	for _, seg := range s.segments {
		s.sortedSegments = append(s.sortedSegments, seg)
	}
	slices.SortFunc(s.sortedSegments, func(a, b *RefCountedSegment) int {
		return cmp.Compare(a.ID(), b.ID())
	})
}

// Clone creates a new Snapshot sharing the same segments (inc refs) and active memtable.
// Tombstones are shallow copied (map copy).
// Note: sortedSegments is NOT cloned; caller must call RebuildSorted() after modifications.
func (s *Snapshot) Clone() *Snapshot {
	newSnap := &Snapshot{
		refs:            1,
		segments:        make(map[model.SegmentID]*RefCountedSegment, len(s.segments)),
		tombstones:      make(map[model.SegmentID]*imetadata.LocalBitmap, len(s.tombstones)),
		active:          s.active,
		activeWatermark: s.activeWatermark,
		pkIndex:         s.pkIndex,
	}

	s.active.IncRef()

	for id, seg := range s.segments {
		seg.IncRef()
		newSnap.segments[id] = seg
	}
	for id, ts := range s.tombstones {
		newSnap.tombstones[id] = ts
	}
	return newSnap
}

// CloneShared creates a new Snapshot sharing the same segments (inc refs) and active memtable.
// Tombstones are shallow copied (map copy).
// Unlike Clone, this shares the segments map and sortedSegments slice, assuming they will NOT be modified.
// This is an optimization for operations like Insert/Delete that do not add/remove segments.
func (s *Snapshot) CloneShared() *Snapshot {
	newSnap := &Snapshot{
		refs:            1,
		segments:        s.segments,       // Shared map
		sortedSegments:  s.sortedSegments, // Shared slice
		tombstones:      make(map[model.SegmentID]*imetadata.LocalBitmap, len(s.tombstones)),
		active:          s.active,
		activeWatermark: s.activeWatermark,
		pkIndex:         s.pkIndex,
	}

	s.active.IncRef()

	// Increment refs for shared segments
	// Prefer iterating sortedSegments if available as it is faster (slice vs map)
	// If sortedSegments is not in sync (e.g. not built), fall back to map.
	if len(s.sortedSegments) == len(s.segments) {
		for _, seg := range s.sortedSegments {
			seg.IncRef()
		}
	} else {
		for _, seg := range s.segments {
			seg.IncRef()
		}
	}

	for id, ts := range s.tombstones {
		newSnap.tombstones[id] = ts
	}
	return newSnap
}
