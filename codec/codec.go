// Package codec centralizes payload and metadata encoding.
//
// Vecgo intentionally treats codec selection as a breaking-change boundary:
// if you change codecs, persisted bytes created by older codecs may no longer decode.
package codec

import "fmt"

// Codec encodes/decodes values.
// Implementations must be safe for concurrent use.
type Codec interface {
	Marshal(v any) ([]byte, error)
	Unmarshal(data []byte, v any) error
	Name() string
}

// ByName returns a built-in codec by its stable name.
//
// This is used for self-describing persistence formats (snapshots/WAL) that store
// the codec name in their header.
func ByName(name string) (Codec, bool) {
	switch name {
	case "json":
		return JSON{}, true
	case "go-json":
		return GoJSON{}, true
	default:
		return nil, false
	}
}

// MustMarshal is a helper for internal tests/benchmarks.
func MustMarshal(c Codec, v any) []byte {
	if c == nil {
		c = Default
	}
	b, err := c.Marshal(v)
	if err != nil {
		panic(fmt.Errorf("codec %s marshal failed: %w", c.Name(), err))
	}
	return b
}
