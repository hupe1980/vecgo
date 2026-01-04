package engine

import (
	"cmp"
	"slices"
	"sync/atomic"

	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/internal/segment/memtable"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// RefCountedSegment wraps a Segment with a reference count.
type RefCountedSegment struct {
	segment.Segment
	refs    int64
	onClose func()
}

func NewRefCountedSegment(seg segment.Segment) *RefCountedSegment {
	return &RefCountedSegment{
		Segment: seg,
		refs:    1, // Initial ref
	}
}

func (r *RefCountedSegment) IncRef() {
	atomic.AddInt64(&r.refs, 1)
}

func (r *RefCountedSegment) DecRef() {
	if atomic.AddInt64(&r.refs, -1) == 0 {
		_ = r.Segment.Close()
		if r.onClose != nil {
			r.onClose()
		}
	}
}

// SetOnClose sets a callback function to be executed when the segment is closed.
// This is typically used to delete the underlying file.
func (r *RefCountedSegment) SetOnClose(f func()) {
	r.onClose = f
}

// Snapshot represents a consistent view of the database.
type Snapshot struct {
	refs           int64
	segments       map[model.SegmentID]*RefCountedSegment
	sortedSegments []*RefCountedSegment // Deterministic iteration order
	tombstones     map[model.SegmentID]*metadata.LocalBitmap
	active         *memtable.MemTable
}

func NewSnapshot(active *memtable.MemTable) *Snapshot {
	return &Snapshot{
		refs:           1,
		segments:       make(map[model.SegmentID]*RefCountedSegment),
		sortedSegments: make([]*RefCountedSegment, 0),
		tombstones:     make(map[model.SegmentID]*metadata.LocalBitmap),
		active:         active,
	}
}

func (s *Snapshot) IncRef() {
	atomic.AddInt64(&s.refs, 1)
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
		refs:       1,
		segments:   make(map[model.SegmentID]*RefCountedSegment, len(s.segments)),
		tombstones: make(map[model.SegmentID]*metadata.LocalBitmap, len(s.tombstones)),
		active:     s.active,
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
