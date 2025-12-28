package index

import (
	"sync"

	"github.com/hupe1980/vecgo/metadata"
)

// InvertedIndex accelerates metadata filtering for common equality/in queries.
//
// Supported operators:
// - OpEqual
// - OpIn (array of Values)
//
// Other operators fall back to scanning + evaluating metadata.FilterSet.
type InvertedIndex struct {
	mu sync.RWMutex

	// key -> valueKey -> ids
	fields map[string]map[string]map[uint32]struct{}
}

func New() *InvertedIndex {
	return &InvertedIndex{fields: make(map[string]map[string]map[uint32]struct{})}
}

func (ix *InvertedIndex) Add(id uint32, doc metadata.Document) {
	if ix == nil || doc == nil {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.addLocked(id, doc)
}

func (ix *InvertedIndex) Remove(id uint32, doc metadata.Document) {
	if ix == nil || doc == nil {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.removeLocked(id, doc)
}

func (ix *InvertedIndex) Update(id uint32, oldDoc, newDoc metadata.Document) {
	if ix == nil {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if oldDoc != nil {
		ix.removeLocked(id, oldDoc)
	}
	if newDoc != nil {
		ix.addLocked(id, newDoc)
	}
}

func (ix *InvertedIndex) addLocked(id uint32, doc metadata.Document) {
	for k, v := range doc {
		vm, ok := ix.fields[k]
		if !ok {
			vm = make(map[string]map[uint32]struct{})
			ix.fields[k] = vm
		}
		vk := v.Key()
		ids, ok := vm[vk]
		if !ok {
			ids = make(map[uint32]struct{})
			vm[vk] = ids
		}
		ids[id] = struct{}{}
	}
}

func (ix *InvertedIndex) removeLocked(id uint32, doc metadata.Document) {
	for k, v := range doc {
		vm, ok := ix.fields[k]
		if !ok {
			continue
		}
		vk := v.Key()
		ids, ok := vm[vk]
		if !ok {
			continue
		}
		delete(ids, id)
		if len(ids) == 0 {
			delete(vm, vk)
		}
		if len(vm) == 0 {
			delete(ix.fields, k)
		}
	}
}

// Compile attempts to compile a FilterSet into a fast membership test using the
// inverted index. If compilation is not possible, ok=false.
func (ix *InvertedIndex) Compile(fs *metadata.FilterSet) (fn func(id uint32) bool, ok bool) {
	if ix == nil || fs == nil || len(fs.Filters) == 0 {
		return nil, false
	}

	ix.mu.RLock()
	defer ix.mu.RUnlock()

	sets := make([]map[uint32]struct{}, 0, len(fs.Filters))

	for _, f := range fs.Filters {
		switch f.Operator {
		case metadata.OpEqual:
			ids := ix.postingsLocked(f.Key, f.Value)
			if ids == nil {
				// Key/value doesn't exist; fast path to always-false.
				return func(uint32) bool { return false }, true
			}
			sets = append(sets, ids)

		case metadata.OpIn:
			arr, ok := f.Value.AsArray()
			if !ok {
				return nil, false
			}
			union := make(map[uint32]struct{})
			for _, vv := range arr {
				ids := ix.postingsLocked(f.Key, vv)
				for id := range ids {
					union[id] = struct{}{}
				}
			}
			if len(union) == 0 {
				return func(uint32) bool { return false }, true
			}
			sets = append(sets, union)

		default:
			return nil, false
		}
	}

	if len(sets) == 0 {
		return nil, false
	}

	// Intersect sets. Start from the smallest to reduce work.
	baseIdx := 0
	baseSize := len(sets[0])
	for i := 1; i < len(sets); i++ {
		if len(sets[i]) < baseSize {
			baseIdx = i
			baseSize = len(sets[i])
		}
	}

	// Copy base set.
	candidates := make(map[uint32]struct{}, baseSize)
	for id := range sets[baseIdx] {
		candidates[id] = struct{}{}
	}

	for i := 0; i < len(sets); i++ {
		if i == baseIdx {
			continue
		}
		other := sets[i]
		for id := range candidates {
			if _, ok := other[id]; !ok {
				delete(candidates, id)
			}
		}
		if len(candidates) == 0 {
			return func(uint32) bool { return false }, true
		}
	}

	return func(id uint32) bool {
		_, ok := candidates[id]
		return ok
	}, true
}

func (ix *InvertedIndex) postingsLocked(key string, v metadata.Value) map[uint32]struct{} {
	vm, ok := ix.fields[key]
	if !ok {
		return nil
	}
	ids, ok := vm[v.Key()]
	if !ok {
		return nil
	}
	return ids
}
