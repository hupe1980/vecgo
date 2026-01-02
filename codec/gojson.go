package codec

import gojson "github.com/goccy/go-json"

// GoJSON is a JSON codec backed by github.com/goccy/go-json.
//
// Persisted formats (snapshots, WAL) store the codec name in their header.
// When opening an existing file, Vecgo selects the codec by name.
type GoJSON struct{}

// Marshal encodes the value to JSON.
func (GoJSON) Marshal(v any) ([]byte, error) { return gojson.Marshal(v) }

// Unmarshal decodes the JSON data into v.
func (GoJSON) Unmarshal(data []byte, v any) error { return gojson.Unmarshal(data, v) }

// Name returns the unique name of the codec ("go-json").
func (GoJSON) Name() string { return "go-json" }

// Append encodes the value to JSON and appends it to dst.
func (GoJSON) Append(dst []byte, v any) ([]byte, error) {
	b, err := gojson.Marshal(v)
	if err != nil {
		return nil, err
	}
	return append(dst, b...), nil
}
