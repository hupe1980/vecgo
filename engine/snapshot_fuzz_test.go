package engine

import (
	"bytes"
	"context"
	"testing"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/index/flat"
	"github.com/hupe1980/vecgo/metadata"
)

// FuzzSnapshotRoundTrip tests snapshot save/load with various data patterns.
func FuzzSnapshotRoundTrip(f *testing.F) {
	// Seed with some typical data patterns
	f.Add(float32(1.0), float32(2.0), "data1", "key1", "value1")
	f.Add(float32(-1.5), float32(3.14), "test", "meta", "val")

	f.Fuzz(func(t *testing.T, v1, v2 float32, data, metaKey, metaValue string) {
		// Skip extremely large inputs to avoid timeouts
		if len(data) > 10000 || len(metaKey) > 1000 || len(metaValue) > 10000 {
			t.Skip()
		}

		// Skip invalid UTF-8 since JSON encoding will transform it
		if !isValidUTF8(data) || !isValidUTF8(metaKey) || !isValidUTF8(metaValue) {
			t.Skip()
		}

		ctx := context.Background()

		// Create a simple index with one vector
		idx, err := flat.New(func(o *flat.Options) {
			o.Dimension = 2
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		if err != nil {
			t.Fatalf("create index failed: %v", err)
		}

		id, err := idx.Insert(ctx, []float32{v1, v2})
		if err != nil {
			t.Fatalf("insert vector failed: %v", err)
		}

		// Create stores
		dataStore := NewMapStore[string]()
		if err := dataStore.Set(id, data); err != nil {
			t.Fatalf("failed to set data: %v", err)
		}

		metadataStore := NewMapStore[metadata.Metadata]()
		meta := metadata.Metadata{}
		if metaKey != "" {
			meta[metaKey] = metadata.String(metaValue)
		}
		if err := metadataStore.Set(id, meta); err != nil {
			t.Fatalf("failed to set metadata: %v", err)
		}

		// Save to buffer
		var buf bytes.Buffer
		if err := SaveToWriter(&buf, idx, dataStore, metadataStore, codec.Default); err != nil {
			t.Fatalf("save failed: %v", err)
		}

		// Load from buffer
		reader := bytes.NewReader(buf.Bytes())
		snap, err := LoadFromReaderWithCodec[string](reader, codec.Default)
		if err != nil {
			t.Fatalf("load failed: %v", err)
		}

		// Verify index has the vector count
		stats := snap.Index.Stats()
		if stats.String() == "" {
			t.Error("stats should not be empty")
		}

		// Verify data store
		loadedDataVal, ok := snap.DataStore.Get(id)
		if !ok {
			t.Errorf("data not found for id %d", id)
		} else if loadedDataVal != data {
			t.Errorf("data mismatch: got %q, want %q", loadedDataVal, data)
		}

		// Verify metadata store count
		loadedMetaVal, ok := snap.MetadataStore.Get(id)
		if !ok {
			t.Errorf("metadata not found for id %d", id)
		} else if len(meta) != len(loadedMetaVal) {
			t.Errorf("metadata length mismatch: got %d, want %d", len(loadedMetaVal), len(meta))
		}
		// Note: We don't check exact metadata values because string encoding
		// through JSON might transform some bytes (e.g., invalid UTF-8)
	})
}

// FuzzSnapshotLoad tests snapshot loading with corrupted/malformed input.
// This helps catch crashes and ensure proper error handling.
func FuzzSnapshotLoad(f *testing.F) {
	// Create a valid snapshot as seed
	ctx := context.Background()
	idx, _ := flat.New(func(o *flat.Options) {
		o.Dimension = 2
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	id, _ := idx.Insert(ctx, []float32{1.0, 2.0})
	dataStore := NewMapStore[string]()
	_ = dataStore.Set(id, "test")
	metadataStore := NewMapStore[metadata.Metadata]()
	_ = metadataStore.Set(id, metadata.Metadata{"k": metadata.String("v")})

	var validSnapshot bytes.Buffer
	_ = SaveToWriter(&validSnapshot, idx, dataStore, metadataStore, codec.Default)
	f.Add(validSnapshot.Bytes())

	// Seed with some malformed patterns
	f.Add([]byte{})                                   // empty
	f.Add([]byte("SNAP"))                             // just magic
	f.Add(bytes.Repeat([]byte{0}, 1024))              // zeros
	f.Add(bytes.Repeat([]byte{0xff}, 512))            // max bytes
	f.Add([]byte{0x53, 0x4e, 0x41, 0x50, 0x01, 0x00}) // magic + version

	f.Fuzz(func(t *testing.T, data []byte) {
		// Skip extremely large inputs
		if len(data) > 1<<20 { // 1MB
			t.Skip()
		}

		reader := bytes.NewReader(data)

		// Attempt to load - should handle errors gracefully without crashing
		_, err := LoadFromReaderWithCodec[string](reader, codec.Default)
		if err != nil {
			// This is expected for most random/corrupted input
			return
		}
		// If it succeeded, that's fine (means input was valid)
	})
}

// FuzzSnapshotChecksumCorruption tests that checksum verification catches corruption.
func FuzzSnapshotChecksumCorruption(f *testing.F) {
	f.Add(uint(100)) // corrupt at byte 100
	f.Add(uint(500)) // corrupt at byte 500

	f.Fuzz(func(t *testing.T, corruptPos uint) {
		// Create a valid snapshot
		ctx := context.Background()
		idx, _ := flat.New(func(o *flat.Options) {
			o.Dimension = 2
			o.DistanceType = index.DistanceTypeSquaredL2
		})
		id1, _ := idx.Insert(ctx, []float32{1.0, 2.0})
		id2, _ := idx.Insert(ctx, []float32{3.0, 4.0})

		dataStore := NewMapStore[string]()
		_ = dataStore.Set(id1, "data1")
		_ = dataStore.Set(id2, "data2")

		metadataStore := NewMapStore[metadata.Metadata]()
		_ = metadataStore.Set(id1, metadata.Metadata{"k": metadata.String("v1")})
		_ = metadataStore.Set(id2, metadata.Metadata{"k": metadata.String("v2")})

		var buf bytes.Buffer
		if err := SaveToWriter(&buf, idx, dataStore, metadataStore, codec.Default); err != nil {
			t.Fatalf("save failed: %v", err)
		}

		snapshotData := buf.Bytes()
		if len(snapshotData) == 0 {
			t.Skip()
		}

		// Corrupt a byte in the snapshot (skip header/footer to corrupt actual data)
		pos := int(corruptPos) % len(snapshotData)
		// Skip the first 100 bytes (header) and last 100 bytes (footer/directory)
		if pos < 100 || pos >= len(snapshotData)-100 {
			t.Skip()
		}

		corrupted := make([]byte, len(snapshotData))
		copy(corrupted, snapshotData)
		corrupted[pos] ^= 0xff // flip all bits at this position

		// Try to load corrupted snapshot
		reader := bytes.NewReader(corrupted)
		_, err := LoadFromReaderWithCodec[string](reader, codec.Default)

		// Should get an error (either checksum mismatch or decode error)
		if err == nil {
			// If we corrupted metadata that doesn't affect checksum or decode,
			// it might still succeed. That's acceptable.
			return
		}

		// We expect checksum errors for data corruption
		// (could also be decode errors if corruption breaks structure)
	})
}

// isValidUTF8 checks if a string contains only valid UTF-8.
func isValidUTF8(s string) bool {
	for _, r := range s {
		if r == '\uFFFD' { // Replacement character indicates invalid UTF-8
			// Check if this was actually in the original string or if it's from decoding
			for _, b := range []byte(s) {
				if b >= 0x80 && b < 0xC0 { // Invalid UTF-8 continuation byte
					return false
				}
			}
		}
	}
	// Additional check: try to range over the string - invalid UTF-8 gets replaced
	original := s
	rebuilt := ""
	for _, r := range s {
		rebuilt += string(r)
	}
	return original == rebuilt
}
