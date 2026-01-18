package memtable

import (
	"testing"

	"github.com/hupe1980/vecgo/internal/segment"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/stretchr/testify/assert"
)

type mockFilter struct {
	allowed map[uint32]bool
}

func (m *mockFilter) Matches(id uint32) bool {
	return m.allowed[id]
}

func (m *mockFilter) MatchesBatch(ids []uint32, out []bool) {
	for i, id := range ids {
		out[i] = m.allowed[id]
	}
}

func (m *mockFilter) AsBitmap() segment.Bitmap                              { return nil }
func (m *mockFilter) MatchesBlock(stats map[string]segment.FieldStats) bool { return true }

func TestColumnarFilterWrapper(t *testing.T) {
	// Setup Columns
	// 0: color=red, price=10
	// 1: color=blue, price=20
	// 2: color=green, price=30
	// 3: deleted/null

	colColor := newStringColumn(4)
	colColor.Append(metadata.String("red"))
	colColor.Append(metadata.String("blue"))
	colColor.Append(metadata.String("green"))
	colColor.Grow(1) // Row 3 empty

	colPrice := newIntColumn(4)
	colPrice.Append(metadata.Int(10))
	colPrice.Append(metadata.Int(20))
	colPrice.Append(metadata.Int(30))
	colPrice.Grow(1) // Row 3 empty

	columns := map[string]Column{
		"color": colColor,
		"price": colPrice,
	}

	t.Run("Basic Filter Equal", func(t *testing.T) {
		// color == "blue"
		f1 := metadata.Filter{
			Key:      "color",
			Operator: metadata.OpEqual,
			Value:    metadata.String("blue"),
		}
		fs1 := metadata.NewFilterSet(f1)

		w := newColumnarFilterWrapper(nil, fs1, columns, 4)

		assert.False(t, w.Matches(0))
		assert.True(t, w.Matches(1))
		assert.False(t, w.Matches(2))
		assert.False(t, w.Matches(3))
		assert.False(t, w.Matches(99)) // Out of bounds
	})

	t.Run("MatchesBatch", func(t *testing.T) {
		f1 := metadata.Filter{
			Key:      "color",
			Operator: metadata.OpEqual,
			Value:    metadata.String("blue"),
		}
		fs1 := metadata.NewFilterSet(f1)
		w := newColumnarFilterWrapper(nil, fs1, columns, 4)

		ids := []uint32{0, 1, 2, 3, 99}
		out := make([]bool, len(ids))
		w.MatchesBatch(ids, out)
		assert.Equal(t, []bool{false, true, false, false, false}, out)
	})

	t.Run("With Parent Filter", func(t *testing.T) {
		// Parent allows 0, 1. Rejects 2.
		parent := &mockFilter{allowed: map[uint32]bool{0: true, 1: true}}

		f1 := metadata.Filter{
			Key:      "color",
			Operator: metadata.OpEqual,
			Value:    metadata.String("blue"),
		}
		fs1 := metadata.NewFilterSet(f1)

		w := newColumnarFilterWrapper(parent, fs1, columns, 4)

		// id=0: Parent True, Meta False ("red" != "blue") -> False
		assert.False(t, w.Matches(0))

		// id=1: Parent True, Meta True ("blue" == "blue") -> True
		assert.True(t, w.Matches(1))

		// id=2: Parent False -> False (even if meta matched?)
		// Meta for 2 is "green" != "blue" anyway.
		assert.False(t, w.Matches(2))
	})

	t.Run("Numeric Filter GreaterThan", func(t *testing.T) {
		// price > 15
		f := metadata.Filter{
			Key:      "price",
			Operator: metadata.OpGreaterThan,
			Value:    metadata.Int(15),
		}
		fs := metadata.NewFilterSet(f)
		w := newColumnarFilterWrapper(nil, fs, columns, 4)

		assert.False(t, w.Matches(0)) // 10
		assert.True(t, w.Matches(1))  // 20
		assert.True(t, w.Matches(2))  // 30
		assert.False(t, w.Matches(3)) // nil
	})

	t.Run("Missing Column", func(t *testing.T) {
		f := metadata.Filter{
			Key:      "missing",
			Operator: metadata.OpEqual,
			Value:    metadata.String("foo"),
		}
		fs := metadata.NewFilterSet(f)
		w := newColumnarFilterWrapper(nil, fs, columns, 4)

		// Always false if column missing
		assert.False(t, w.Matches(0))
		assert.False(t, w.Matches(1))
	})

	t.Run("OpIn String Filter", func(t *testing.T) {
		// color IN ("red", "green")
		f := metadata.Filter{
			Key:      "color",
			Operator: metadata.OpIn,
			Value:    metadata.Array([]metadata.Value{metadata.String("red"), metadata.String("green")}),
		}
		fs := metadata.NewFilterSet(f)
		w := newColumnarFilterWrapper(nil, fs, columns, 4)

		assert.True(t, w.Matches(0))  // "red" ✓
		assert.False(t, w.Matches(1)) // "blue" ✗
		assert.True(t, w.Matches(2))  // "green" ✓
		assert.False(t, w.Matches(3)) // nil ✗
	})

	t.Run("OpIn Int Filter", func(t *testing.T) {
		// price IN (10, 30)
		f := metadata.Filter{
			Key:      "price",
			Operator: metadata.OpIn,
			Value:    metadata.Array([]metadata.Value{metadata.Int(10), metadata.Int(30)}),
		}
		fs := metadata.NewFilterSet(f)
		w := newColumnarFilterWrapper(nil, fs, columns, 4)

		assert.True(t, w.Matches(0))  // 10 ✓
		assert.False(t, w.Matches(1)) // 20 ✗
		assert.True(t, w.Matches(2))  // 30 ✓
		assert.False(t, w.Matches(3)) // nil ✗
	})

	t.Run("OpIn Single Element", func(t *testing.T) {
		// price IN (20) - single element should work
		f := metadata.Filter{
			Key:      "price",
			Operator: metadata.OpIn,
			Value:    metadata.Array([]metadata.Value{metadata.Int(20)}),
		}
		fs := metadata.NewFilterSet(f)
		w := newColumnarFilterWrapper(nil, fs, columns, 4)

		assert.False(t, w.Matches(0)) // 10 ✗
		assert.True(t, w.Matches(1))  // 20 ✓
		assert.False(t, w.Matches(2)) // 30 ✗
	})

	t.Run("OpIn Empty Array", func(t *testing.T) {
		// price IN () - empty array matches nothing
		f := metadata.Filter{
			Key:      "price",
			Operator: metadata.OpIn,
			Value:    metadata.Array([]metadata.Value{}),
		}
		fs := metadata.NewFilterSet(f)
		w := newColumnarFilterWrapper(nil, fs, columns, 4)

		assert.False(t, w.Matches(0))
		assert.False(t, w.Matches(1))
		assert.False(t, w.Matches(2))
	})
}
