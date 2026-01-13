package memtable

import (
	"sync"
	"sync/atomic"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

const (
	pageSize = 65536 // 2^16
	pageMask = pageSize - 1
	pageBits = 16
)

// PagedIDStore stores IDs in chunks.
type PagedIDStore struct {
	mu    sync.Mutex
	pages atomic.Pointer[[]*[pageSize]model.ID]
	count atomic.Uint64
}

func NewPagedIDStore() *PagedIDStore {
	s := &PagedIDStore{}
	// Init empty pages
	p := make([]*[pageSize]model.ID, 0, 16)
	s.pages.Store(&p)
	return s
}

func (s *PagedIDStore) Count() uint64 {
	return s.count.Load()
}

func (s *PagedIDStore) Get(rowID uint32) (model.ID, bool) {
	pageIdx := int(rowID) >> pageBits
	offset := int(rowID) & pageMask

	pagesPtr := s.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		return 0, false
	}
	p := pages[pageIdx]
	if p == nil {
		return 0, false
	}
	// Bound check within page? offset < pageSize always true by mask.
	// Check against count?
	if uint64(rowID) >= s.count.Load() {
		return 0, false
	}

	id := p[offset]
	return id, true
}

// Append adds an ID. Thread-safe via external lock (MemTable.mu) or internal lock?
// We'll use internal lock for growth, but Append is usually called under MemTable.mu anyway.
// But to avoid O(N), we must not copy the whole data.
func (s *PagedIDStore) Append(id model.ID) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := s.count.Load()
	pageIdx := int(idx) >> pageBits
	offset := int(idx) & pageMask

	pagesPtr := s.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		// Grow pages slice
		newCap := len(pages) * 2
		if newCap == 0 {
			newCap = 16
		}
		newPages := make([]*[pageSize]model.ID, len(pages), newCap)
		copy(newPages, pages)

		// Add new page
		newPages = append(newPages, new([pageSize]model.ID))
		s.pages.Store(&newPages)
		pages = newPages // Update local view
	}
	// Note: pages[pageIdx] == nil is unreachable with strict sequential append pattern

	// Write
	pages[pageIdx][offset] = id
	s.count.Store(idx + 1)
}

// PagedMetaStore stores metadata in chunks.
type PagedMetaStore struct {
	mu    sync.Mutex
	pages atomic.Pointer[[]*[pageSize]metadata.InternedDocument]
	count atomic.Uint64
}

func NewPagedMetaStore() *PagedMetaStore {
	s := &PagedMetaStore{}
	p := make([]*[pageSize]metadata.InternedDocument, 0, 16)
	s.pages.Store(&p)
	return s
}

func (s *PagedMetaStore) Count() uint64 {
	return s.count.Load()
}

func (s *PagedMetaStore) Get(rowID uint32) (metadata.InternedDocument, bool) {
	pageIdx := int(rowID) >> pageBits
	offset := int(rowID) & pageMask

	pagesPtr := s.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		return nil, false
	}
	p := pages[pageIdx]
	if p == nil {
		return nil, false
	}
	if uint64(rowID) >= s.count.Load() {
		return nil, false
	}
	return p[offset], true
}

func (s *PagedMetaStore) Append(md metadata.InternedDocument) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := s.count.Load()
	pageIdx := int(idx) >> pageBits
	offset := int(idx) & pageMask

	pagesPtr := s.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		newCap := len(pages) * 2
		if newCap == 0 {
			newCap = 16
		}
		newPages := make([]*[pageSize]metadata.InternedDocument, len(pages), newCap)
		copy(newPages, pages)
		newPages = append(newPages, new([pageSize]metadata.InternedDocument))
		s.pages.Store(&newPages)
		pages = newPages
	}
	pages[pageIdx][offset] = md
	s.count.Store(idx + 1)
}

// PagedPayloadStore stores payloads in chunks.
type PagedPayloadStore struct {
	mu    sync.Mutex
	pages atomic.Pointer[[]*[pageSize][]byte]
	count atomic.Uint64
}

func NewPagedPayloadStore() *PagedPayloadStore {
	s := &PagedPayloadStore{}
	p := make([]*[pageSize][]byte, 0, 16)
	s.pages.Store(&p)
	return s
}

func (s *PagedPayloadStore) Count() uint64 {
	return s.count.Load()
}

func (s *PagedPayloadStore) Get(rowID uint32) ([]byte, bool) {
	pageIdx := int(rowID) >> pageBits
	offset := int(rowID) & pageMask

	pagesPtr := s.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) || pages[pageIdx] == nil || uint64(rowID) >= s.count.Load() {
		return nil, false
	}
	return pages[pageIdx][offset], true
}

func (s *PagedPayloadStore) Append(payload []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := s.count.Load()
	pageIdx := int(idx) >> pageBits
	offset := int(idx) & pageMask

	pagesPtr := s.pages.Load()
	pages := *pagesPtr

	if pageIdx >= len(pages) {
		newCap := len(pages) * 2
		if newCap == 0 {
			newCap = 16
		}
		newPages := make([]*[pageSize][]byte, len(pages), newCap)
		copy(newPages, pages)
		newPages = append(newPages, new([pageSize][]byte))
		s.pages.Store(&newPages)
		pages = newPages
	}
	pages[pageIdx][offset] = payload
	s.count.Store(idx + 1)
}
