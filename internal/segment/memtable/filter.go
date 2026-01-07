package memtable

import (
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
)

// metadataFilterWrapper combines a base filter (e.g. tombstones) with a user metadata filter.
type metadataFilterWrapper struct {
	parent segment.Filter
	meta   *metadata.FilterSet
	docs   []metadata.InternedDocument
}

func (w *metadataFilterWrapper) Matches(id uint32) bool {
	if w.parent != nil && !w.parent.Matches(id) {
		return false
	}
	if int(id) >= len(w.docs) {
		return false
	}
	doc := w.docs[int(id)]
	if doc == nil {
		return false
	}
	return w.meta.MatchesInterned(doc)
}

func (w *metadataFilterWrapper) MatchesBatch(ids []uint32, out []bool) {
	// Naive implementation
	for i, id := range ids {
		out[i] = w.Matches(id)
	}
}

func (w *metadataFilterWrapper) AsBitmap() segment.Bitmap {
	return nil
}

func (w *metadataFilterWrapper) MatchesBlock(stats map[string]segment.FieldStats) bool {
	// Conservative: match everything if we can't check efficiently
	return true
}
