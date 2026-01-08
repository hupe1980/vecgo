package memtable

import (
	"fmt"

	"github.com/hupe1980/vecgo/metadata"
)

type Column interface {
	Grow(n int)
	Append(v metadata.Value)
	Set(i int, v metadata.Value)
	Get(i int) (metadata.Value, bool)
	Len() int
}

type intColumn struct {
	data  []int64
	valid []bool
}

func newIntColumn(cap int) *intColumn {
	return &intColumn{
		data:  make([]int64, 0, cap),
		valid: make([]bool, 0, cap),
	}
}

func (c *intColumn) Len() int { return len(c.data) }

func (c *intColumn) Grow(n int) {
	// Pad with nulls
	for i := 0; i < n; i++ {
		c.data = append(c.data, 0)
		c.valid = append(c.valid, false)
	}
}

func (c *intColumn) Append(v metadata.Value) {
	if v.Kind == metadata.KindInt {
		val, _ := v.AsInt64()
		c.data = append(c.data, val)
		c.valid = append(c.valid, true)
	} else {
		c.data = append(c.data, 0)
		c.valid = append(c.valid, false)
	}
}

func (c *intColumn) Set(i int, v metadata.Value) {
	if i >= len(c.data) {
		return // Should grow first
	}
	if v.Kind == metadata.KindInt {
		val, _ := v.AsInt64()
		c.data[i] = val
		c.valid[i] = true
	} else {
		c.valid[i] = false
	}
}

func (c *intColumn) Get(i int) (metadata.Value, bool) {
	if i >= len(c.data) || !c.valid[i] {
		return metadata.Value{}, false
	}
	return metadata.Int(c.data[i]), true
}

type floatColumn struct {
	data  []float64
	valid []bool
}

func newFloatColumn(cap int) *floatColumn {
	return &floatColumn{
		data:  make([]float64, 0, cap),
		valid: make([]bool, 0, cap),
	}
}

func (c *floatColumn) Len() int { return len(c.data) }

func (c *floatColumn) Grow(n int) {
	for i := 0; i < n; i++ {
		c.data = append(c.data, 0)
		c.valid = append(c.valid, false)
	}
}

func (c *floatColumn) Append(v metadata.Value) {
	if v.Kind == metadata.KindFloat {
		val, _ := v.AsFloat64()
		c.data = append(c.data, val)
		c.valid = append(c.valid, true)
	} else if v.Kind == metadata.KindInt {
		// Allow promotion
		val, _ := v.AsInt64()
		c.data = append(c.data, float64(val))
		c.valid = append(c.valid, true)
	} else {
		c.data = append(c.data, 0)
		c.valid = append(c.valid, false)
	}
}

func (c *floatColumn) Set(i int, v metadata.Value) {
	if i >= len(c.data) {
		return
	}
	if v.Kind == metadata.KindFloat {
		val, _ := v.AsFloat64()
		c.data[i] = val
		c.valid[i] = true
	} else if v.Kind == metadata.KindInt {
		val, _ := v.AsInt64()
		c.data[i] = float64(val)
		c.valid[i] = true
	} else {
		c.valid[i] = false
	}
}

func (c *floatColumn) Get(i int) (metadata.Value, bool) {
	if i >= len(c.data) || !c.valid[i] {
		return metadata.Value{}, false
	}
	return metadata.Float(c.data[i]), true
}

type stringColumn struct {
	data  []string
	valid []bool
}

func newStringColumn(cap int) *stringColumn {
	return &stringColumn{
		data:  make([]string, 0, cap),
		valid: make([]bool, 0, cap),
	}
}

func (c *stringColumn) Len() int { return len(c.data) }

func (c *stringColumn) Grow(n int) {
	for i := 0; i < n; i++ {
		c.data = append(c.data, "")
		c.valid = append(c.valid, false)
	}
}

func (c *stringColumn) Append(v metadata.Value) {
	if v.Kind == metadata.KindString {
		val, _ := v.AsString()
		c.data = append(c.data, val)
		c.valid = append(c.valid, true)
	} else {
		c.data = append(c.data, "")
		c.valid = append(c.valid, false)
	}
}

func (c *stringColumn) Set(i int, v metadata.Value) {
	if i >= len(c.data) {
		return
	}
	if v.Kind == metadata.KindString {
		val, _ := v.AsString()
		c.data[i] = val
		c.valid[i] = true
	} else {
		c.valid[i] = false
	}
}

func (c *stringColumn) Get(i int) (metadata.Value, bool) {
	if i >= len(c.data) || !c.valid[i] {
		return metadata.Value{}, false
	}
	return metadata.String(c.data[i]), true
}

type boolColumn struct {
	data  []bool
	valid []bool
}

func newBoolColumn(cap int) *boolColumn {
	return &boolColumn{
		data:  make([]bool, 0, cap),
		valid: make([]bool, 0, cap),
	}
}

func (c *boolColumn) Len() int { return len(c.data) }

func (c *boolColumn) Grow(n int) {
	for i := 0; i < n; i++ {
		c.data = append(c.data, false)
		c.valid = append(c.valid, false)
	}
}

func (c *boolColumn) Append(v metadata.Value) {
	if v.Kind == metadata.KindBool {
		val, _ := v.AsBool()
		c.data = append(c.data, val)
		c.valid = append(c.valid, true)
	} else {
		c.data = append(c.data, false)
		c.valid = append(c.valid, false)
	}
}

func (c *boolColumn) Set(i int, v metadata.Value) {
	if i >= len(c.data) {
		return
	}
	if v.Kind == metadata.KindBool {
		val, _ := v.AsBool()
		c.data[i] = val
		c.valid[i] = true
	} else {
		c.valid[i] = false
	}
}

func (c *boolColumn) Get(i int) (metadata.Value, bool) {
	if i >= len(c.data) || !c.valid[i] {
		return metadata.Value{}, false
	}
	return metadata.Bool(c.data[i]), true
}

// createColumn creates a column based on the value kind.
func createColumn(kind metadata.Kind, capacity int) (Column, error) {
	switch kind {
	case metadata.KindInt:
		return newIntColumn(capacity), nil
	case metadata.KindFloat:
		return newFloatColumn(capacity), nil
	case metadata.KindString:
		return newStringColumn(capacity), nil
	case metadata.KindBool:
		return newBoolColumn(capacity), nil
	default:
		// Fallback or error? For now, ignore arrays/objects in columnar optimization
		return nil, fmt.Errorf("unsupported columnar type: %v", kind)
	}
}
