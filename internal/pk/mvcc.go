package pk

import (
	"iter"
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/model"
)

const (
	pageBits = 16
	pageSize = 1 << pageBits // 65536
	pageMask = pageSize - 1
)

// versionPool pools version structs to reduce allocations.
var versionPool = sync.Pool{
	New: func() any {
		return &version{}
	},
}

func newVersion(lsn uint64, loc model.Location, deleted bool) *version {
	v := versionPool.Get().(*version)
	v.lsn = lsn
	v.location = loc
	v.deleted = deleted
	v.next = nil
	return v
}

// Index is a concurrent, paged, MVCC primary key index optimized for sequential IDs.
// It uses a dynamic array of fixed-size pages to provide O(1) access without hashing overhead.
type Index struct {
	mu    sync.RWMutex // Protects pages slice growth
	pages atomic.Pointer[[]*page]
	count atomic.Int64
}

type page struct {
	entries [pageSize]entry
}

type entry struct {
	head atomic.Pointer[version]
}

type version struct {
	lsn      uint64
	location model.Location
	deleted  bool
	next     *version
}

// New creates a new MVCC Index.
func New() *Index {
	idx := &Index{}
	// Start with empty pages
	p := make([]*page, 0, 16)
	idx.pages.Store(&p)
	return idx
}

// ensurePageExists guarantees that the page for the given pageIdx exists.
// Returns the page slice (snapshot) which is safe to read.
func (idx *Index) ensurePageExists(pageIdx int) []*page {
	// Fast path: check atomic load
	pagesPtr := idx.pages.Load()
	pages := *pagesPtr
	if pageIdx < len(pages) {
		return pages
	}

	// Slow path: grow
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Reload under lock
	pagesPtr = idx.pages.Load()
	pages = *pagesPtr

	if pageIdx < len(pages) {
		return pages
	}

	// Calculate new size
	newCap := len(pages)
	if newCap == 0 {
		newCap = 16
	}
	for newCap <= pageIdx {
		newCap *= 2
	}

	newPages := make([]*page, newCap)
	copy(newPages, pages)

	// Allocate missing pages up to pageIdx
	// (Actually we just need to allocate up to pageIdx,
	// but simplest is to just fill what we need?
	// Or simplistic: just ensure [pageIdx] is non-nil).
	// But we need to fill gaps if any (though sequential IDs imply filling).
	// Let's just fill up to `newCap` to avoid repeated allocs?
	// No, allocate strictly what is needed but extend slice?
	// To minimize allocs, we just ensure newPages[pageIdx] is valid?
	// No, `rows` logic usually fills sequentially.
	// Let's just allocate the specific page requested and any holes.

	for i := len(pages); i <= pageIdx; i++ {
		newPages[i] = &page{}
	}

	// Wrap and store
	// Note: We might be keeping length as pageIdx + 1?
	// Or newCap?
	// Slice length should be pageIdx + 1. Capacity can be newCap.
	finalSlice := newPages[:pageIdx+1]

	idx.pages.Store(&finalSlice)
	return finalSlice
}

// Get returns the location valid at the given snapshot LSN.
func (idx *Index) Get(id model.ID, snapshotLSN uint64) (model.Location, bool) {
	if id == 0 {
		return model.Location{}, false
	}

	i := uint64(id)
	pageIdx := int(i) >> pageBits
	offset := int(i) & pageMask

	pagesPtr := idx.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		return model.Location{}, false
	}
	p := pages[pageIdx]
	if p == nil {
		return model.Location{}, false
	}

	e := &p.entries[offset]
	curr := e.head.Load()

	for curr != nil {
		if curr.lsn <= snapshotLSN {
			if curr.deleted {
				return model.Location{}, false
			}
			return curr.location, true
		}
		curr = curr.next
	}

	return model.Location{}, false
}

// Upsert adds a new version and returns the previous active location and true if it existed.
func (idx *Index) Upsert(id model.ID, loc model.Location, lsn uint64) (model.Location, bool) {
	if id == 0 {
		return model.Location{}, false
	}

	i := uint64(id)
	pageIdx := int(i) >> pageBits
	offset := int(i) & pageMask

	// Ensure page exists
	pages := idx.ensurePageExists(pageIdx)
	p := pages[pageIdx]
	e := &p.entries[offset]

	for {
		head := e.head.Load()
		newV := newVersion(lsn, loc, false)

		isNewEntry := (head == nil)

		// Case 1: Empty list or Prepend (newV is newer)
		if head == nil || head.lsn < lsn {
			newV.next = head
			if e.head.CompareAndSwap(head, newV) {
				if isNewEntry {
					idx.count.Add(1)
					return model.Location{}, false
				}
				if head.deleted {
					return model.Location{}, false
				}
				return head.location, true
			}
			versionPool.Put(newV)
			continue
		}

		// Case 2: Update Head (lsn == head.lsn)
		if head.lsn == lsn {
			newV.next = head.next
			oldLoc, oldDel := head.location, head.deleted

			if e.head.CompareAndSwap(head, newV) {
				// Success
				return oldLoc, !oldDel
			}
			versionPool.Put(newV)
			continue
		}

		// Case 3: Insert in Middle (lsn < head.lsn)
		// COW path
		newHead := copyChainWithInsert(head, newV)
		if e.head.CompareAndSwap(head, newHead) {
			return model.Location{}, false
		}
		// CAS failed, retry
		// Note: newHead and its clones are leaked to GC if logic fails.
		// Given robust GC, this is acceptable for rare contention.
		continue
	}
}

func copyChainWithInsert(head *version, newV *version) *version {
	// Stack to hold nodes we need to clone
	// Fixed size stack optimization for typical depths?
	// Dynamic slice is fine.
	var stack []*version
	curr := head
	for curr != nil && curr.lsn > newV.lsn {
		stack = append(stack, curr)
		curr = curr.next
	}

	var tail *version
	if curr != nil && curr.lsn == newV.lsn {
		// Overwrite curr
		newV.next = curr.next
		tail = newV
	} else {
		// Prepend newV to curr
		newV.next = curr
		tail = newV
	}

	// Unwind stack
	for i := len(stack) - 1; i >= 0; i-- {
		orig := stack[i]
		clone := newVersion(orig.lsn, orig.location, orig.deleted)
		clone.next = tail
		tail = clone
	}
	return tail
}

// Delete adds a tombstone.
func (idx *Index) Delete(id model.ID, lsn uint64) (model.Location, bool) {
	if id == 0 {
		return model.Location{}, false
	}

	i := uint64(id)
	pageIdx := int(i) >> pageBits
	offset := int(i) & pageMask

	pagesPtr := idx.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		return model.Location{}, false
	}
	p := pages[pageIdx]
	if p == nil {
		return model.Location{}, false
	}
	e := &p.entries[offset]

	for {
		head := e.head.Load()
		if head == nil {
			return model.Location{}, false
		}

		newV := newVersion(lsn, model.Location{}, true)

		// Case 1: Prepend (newV is newer)
		if head.lsn < lsn {
			newV.next = head
			if e.head.CompareAndSwap(head, newV) {
				if !head.deleted {
					idx.count.Add(-1)
				}
				return head.location, !head.deleted
			}
			versionPool.Put(newV)
			continue
		}

		// Case 2: Update Head (lsn == head.lsn)
		if head.lsn == lsn {
			if head.deleted {
				versionPool.Put(newV)
				return model.Location{}, false
			}
			newV.next = head.next
			if e.head.CompareAndSwap(head, newV) {
				idx.count.Add(-1)
				return head.location, true
			}
			versionPool.Put(newV)
			continue
		}

		// Case 3: Insert in Middle (lsn < head.lsn)
		// COW path for deletion of OLD version?
		// "Deleting" an old version usually means marking it as deleted in history.
		// If we are deleting at an old LSN, we are saying "at time T_old, it was deleted".
		// This should not affect current head count if head is newer.
		newHead := copyChainWithInsert(head, newV)
		if e.head.CompareAndSwap(head, newHead) {
			// Count update: only if we deleted the currently active head?
			// But head is NOT deleted.
			// We effectively inserted a hole in history.
			return model.Location{}, false
		}
		continue
	}
}

// Count returns approximate number of active keys.
func (idx *Index) Count() int {
	return int(idx.count.Load())
}

// MaxID returns the highest ID currently in the index.
// Used for recovery to initialize the auto-increment counter.
func (idx *Index) MaxID() uint64 {
	pagesPtr := idx.pages.Load()
	pages := *pagesPtr

	for i := len(pages) - 1; i >= 0; i-- {
		p := pages[i]
		if p == nil {
			continue
		}

		// Found highest page, scan for highest entry
		baseID := uint64(i) << pageBits
		for j := pageSize - 1; j >= 0; j-- {
			if p.entries[j].head.Load() != nil {
				return baseID + uint64(j)
			}
		}
	}
	return 0
}

// Scan returns an iterator over all items visible at snapshotLSN.
func (idx *Index) Scan(snapshotLSN uint64) iter.Seq2[model.ID, model.Location] {
	return func(yield func(model.ID, model.Location) bool) {
		pagesPtr := idx.pages.Load()
		pages := *pagesPtr

		for pIdx, p := range pages {
			if p == nil {
				continue
			}
			baseID := uint64(pIdx) << pageBits

			// Iterate efficiently
			for off := 0; off < pageSize; off++ {
				e := &p.entries[off]

				// Quick check without lock
				head := e.head.Load()
				if head == nil {
					continue
				}

				curr := head
				var foundLoc model.Location
				var found bool

				for curr != nil {
					if curr.lsn <= snapshotLSN {
						if !curr.deleted {
							foundLoc = curr.location
							found = true
						}
						break
					}
					curr = curr.next
				}

				if found {
					id := model.ID(baseID + uint64(off))
					if id == 0 {
						continue
					}
					if !yield(id, foundLoc) {
						return
					}
				}
			}
		}
	}
}
