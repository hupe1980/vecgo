package hnsw

import (
	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/model"
)

type mockFilter func(model.RowID) bool

func (f mockFilter) Matches(id uint32) bool {
	return f(model.RowID(id))
}

func (f mockFilter) MatchesBatch(ids []uint32, out []bool) {
	for i, id := range ids {
		out[i] = f(model.RowID(id))
	}
}

func (f mockFilter) AsBitmap() segment.Bitmap { return nil }

func (f mockFilter) MatchesBlock(stats map[string]segment.FieldStats) bool { return true }
