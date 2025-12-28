package codec

import (
	"encoding/json"
)

// JSON is the standard-library JSON codec.
//
// Notes:
// - For metadata (map-like structures), JSON is stable and portable.
// - For arbitrary user payloads, JSON works for typical structs/maps/slices.
// - Time, complex numbers, funcs, channels, etc may not be supported.
//
// If you need custom encoding (e.g. protobuf/msgpack), implement Codec and set
// it on Vecgo/WAL/snapshots where supported.
//
// Performance note:
//   - If you need the most portable/lowest-dependency option, use JSON.
//   - Vecgo's default codec may change over time; persisted data always records
//     the codec name so it can be validated on load.
type JSON struct{}

func (JSON) Marshal(v any) ([]byte, error)      { return json.Marshal(v) }
func (JSON) Unmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }
func (JSON) Name() string                       { return "json" }

// Default is the default codec used by the library.
//
// NOTE: This affects newly-created snapshots/WALs. Existing persisted files are
// self-describing (they store the codec name in their header) and are opened by
// selecting the appropriate codec by name.
var Default Codec = GoJSON{}
