package wal

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
)

// FuzzWALEntry tests the WAL encoding/decoding with arbitrary entry data.
// It ensures that any entry can be written and read back correctly.
func FuzzWALEntry(f *testing.F) {
	// Seed with some typical patterns
	f.Add(uint32(1), float32(1.0), float32(2.0), []byte("data1"), []byte(`{"key":"value"}`))
	f.Add(uint32(0), float32(0.0), float32(0.0), []byte(""), []byte("{}"))
	f.Add(uint32(999), float32(-1.5), float32(3.14), []byte("test"), []byte(`{}`))

	f.Fuzz(func(t *testing.T, id uint32, v1, v2 float32, data, metaJSON []byte) {
		// Skip extremely large inputs to avoid timeout
		if len(data) > 100000 || len(metaJSON) > 10000 {
			t.Skip()
		}

		// Skip ID 0 as it might not be valid
		if id == 0 {
			t.Skip()
		}

		tmpDir := t.TempDir()

		// Create WAL
		wal, err := New(func(o *Options) {
			o.Path = tmpDir
		})
		if err != nil {
			t.Fatalf("create WAL failed: %v", err)
		}

		// Parse metadata if valid JSON
		meta := metadata.Metadata{}
		if len(metaJSON) > 0 {
			// Try to unmarshal, but don't fail if it's invalid JSON
			_ = json.Unmarshal(metaJSON, &meta)
		}

		// Write entry
		if err := wal.LogInsert(id, []float32{v1, v2}, data, meta); err != nil {
			_ = wal.Close()
			t.Fatalf("LogInsert failed: %v", err)
		}

		if err := wal.Close(); err != nil {
			t.Fatalf("close failed: %v", err)
		}

		// Read it back
		walRead, err := New(func(o *Options) {
			o.Path = tmpDir
		})
		if err != nil {
			t.Fatalf("reopen WAL failed: %v", err)
		}
		defer walRead.Close()

		// Replay committed operations
		var readOps []struct {
			id     uint32
			vector []float32
			data   []byte
		}

		if err := walRead.ReplayCommitted(func(entry Entry) error {
			if entry.Type == OpInsert {
				readOps = append(readOps, struct {
					id     uint32
					vector []float32
					data   []byte
				}{entry.ID, entry.Vector, entry.Data})
			}
			return nil
		}); err != nil {
			t.Fatalf("replay failed: %v", err)
		}

		if len(readOps) != 1 {
			t.Fatalf("expected 1 operation, got %d", len(readOps))
		}

		// Verify the data matches
		readOp := readOps[0]
		if readOp.id != id {
			t.Errorf("ID mismatch: got %v, want %v", readOp.id, id)
		}
		if len(readOp.vector) != 2 || readOp.vector[0] != v1 || readOp.vector[1] != v2 {
			t.Errorf("vector mismatch: got %v, want [%v %v]", readOp.vector, v1, v2)
		}
		if !bytes.Equal(readOp.data, data) {
			t.Errorf("data mismatch: len=%d vs %d", len(readOp.data), len(data))
		}
	})
}

// FuzzWALReplay tests WAL replay with corrupted/malformed files.
// This helps catch crashes from corrupted WAL files.
func FuzzWALReplay(f *testing.F) {
	// Create a valid WAL file as seed
	tmpDir := f.TempDir()
	wal, _ := New(func(o *Options) {
		o.Path = tmpDir
	})
	_ = wal.LogInsert(1, []float32{1.0, 2.0}, []byte("test"), nil)
	_ = wal.Close()

	walPath := filepath.Join(tmpDir, "vecgo.wal")
	validData, _ := os.ReadFile(walPath)
	f.Add(validData)

	// Seed with some malformed patterns
	f.Add([]byte{})                        // empty
	f.Add([]byte("WLOG"))                  // just magic
	f.Add(bytes.Repeat([]byte{0}, 1024))   // zeros
	f.Add(bytes.Repeat([]byte{0xff}, 512)) // max bytes

	f.Fuzz(func(t *testing.T, data []byte) {
		// Skip extremely large inputs
		if len(data) > 1<<20 { // 1MB
			t.Skip()
		}

		// Write corrupted data to a file
		tmpDir := t.TempDir()
		tmpPath := filepath.Join(tmpDir, "vecgo.wal")
		if err := os.WriteFile(tmpPath, data, 0644); err != nil {
			t.Fatalf("write file failed: %v", err)
		}

		// Try to open and replay - should handle errors gracefully
		wal, err := New(func(o *Options) {
			o.Path = tmpDir
		})
		if err != nil {
			// Expected for most corrupted data
			return
		}
		defer wal.Close()

		// Attempt replay - should not crash
		_ = wal.ReplayCommitted(func(entry Entry) error {
			return nil
		})
	})
}

// FuzzWALMultipleOperations tests WAL with various operation sequences.
func FuzzWALMultipleOperations(f *testing.F) {
	f.Add(uint8(1), uint32(100))
	f.Add(uint8(5), uint32(1))

	f.Fuzz(func(t *testing.T, opCount uint8, baseID uint32) {
		// Limit operation count
		if opCount == 0 || opCount > 50 {
			t.Skip()
		}

		// Skip baseID 0
		if baseID == 0 {
			baseID = 1
		}

		tmpDir := t.TempDir()
		wal, err := New(func(o *Options) {
			o.Path = tmpDir
		})
		if err != nil {
			t.Fatalf("create failed: %v", err)
		}

		// Write multiple operations
		for i := uint8(0); i < opCount; i++ {
			if err := wal.LogInsert(
				baseID+uint32(i),
				[]float32{float32(i), float32(i + 1)},
				[]byte{byte(i)},
				nil,
			); err != nil {
				_ = wal.Close()
				t.Fatalf("LogInsert %d failed: %v", i, err)
			}
		}

		if err := wal.Close(); err != nil {
			t.Fatalf("close failed: %v", err)
		}

		// Read back and verify count
		walRead, err := New(func(o *Options) {
			o.Path = tmpDir
		})
		if err != nil {
			t.Fatalf("reopen failed: %v", err)
		}
		defer walRead.Close()

		count := 0
		if err := walRead.ReplayCommitted(func(entry Entry) error {
			if entry.Type == OpInsert {
				count++
			}
			return nil
		}); err != nil {
			t.Fatalf("replay failed: %v", err)
		}

		if count != int(opCount) {
			t.Errorf("operation count mismatch: got %d, want %d", count, opCount)
		}
	})
}
