package wal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
)

func TestWAL(t *testing.T) {
	dir := t.TempDir()

	// Create WAL
	wal, err := New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()

	// Test Insert
	data := encodeData(t, "test-data")
	meta := metadata.Metadata{"key": metadata.String("value")}

	err = wal.LogInsert(1, []float32{1.0, 2.0, 3.0}, data, meta)
	if err != nil {
		t.Fatalf("LogInsert failed: %v", err)
	}

	// Test Update
	err = wal.LogUpdate(1, []float32{1.1, 2.1, 3.1}, encodeData(t, "updated"), nil)
	if err != nil {
		t.Fatalf("LogUpdate failed: %v", err)
	}

	// Test Delete
	err = wal.LogDelete(2)
	if err != nil {
		t.Fatalf("LogDelete failed: %v", err)
	}

	// Verify entries
	count, err := wal.Len()
	if err != nil {
		t.Fatalf("Len failed: %v", err)
	}
	// Each operation is written as Prepare+Commit.
	if count != 6 {
		t.Errorf("Expected 6 entries, got %d", count)
	}
}

func TestWALReplay(t *testing.T) {
	dir := t.TempDir()

	// Create WAL and write entries
	wal, err := New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}

	operations := []struct {
		id     uint64
		vector []float32
		data   string
	}{
		{1, []float32{1.0, 0.0, 0.0}, "data1"},
		{2, []float32{0.0, 1.0, 0.0}, "data2"},
		{3, []float32{0.0, 0.0, 1.0}, "data3"},
	}

	for _, op := range operations {
		err := wal.LogInsert(op.id, op.vector, encodeData(t, op.data), nil)
		if err != nil {
			t.Fatalf("LogInsert failed: %v", err)
		}
	}

	wal.Close()

	// Reopen and replay
	wal, err = New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to reopen WAL: %v", err)
	}
	defer wal.Close()

	replayed := []Entry{}
	err = wal.ReplayCommitted(func(entry Entry) error {
		replayed = append(replayed, entry)
		return nil
	})
	if err != nil {
		t.Fatalf("ReplayCommitted failed: %v", err)
	}

	if len(replayed) != 3 {
		t.Errorf("Expected 3 replayed entries, got %d", len(replayed))
	}

	for i, entry := range replayed {
		if entry.ID != operations[i].id {
			t.Errorf("Entry %d: expected ID %d, got %d", i, operations[i].id, entry.ID)
		}
		if entry.Type != OpInsert {
			t.Errorf("Entry %d: expected OpInsert, got %v", i, entry.Type)
		}
	}
}

func TestWALReplayCommittedIgnoresUncommittedPrepares(t *testing.T) {
	dir := t.TempDir()

	w, err := New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}

	// Prepare without commit (should be ignored).
	if err := w.LogPrepareInsert(1, []float32{1, 0, 0}, encodeData(t, "data1"), nil); err != nil {
		t.Fatalf("LogPrepareInsert failed: %v", err)
	}

	// Prepare + commit (should be applied).
	if err := w.LogPrepareInsert(2, []float32{0, 1, 0}, encodeData(t, "data2"), nil); err != nil {
		t.Fatalf("LogPrepareInsert failed: %v", err)
	}
	if err := w.LogCommitInsert(2); err != nil {
		t.Fatalf("LogCommitInsert failed: %v", err)
	}

	_ = w.Close()

	// Reopen and replay committed
	w, err = New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to reopen WAL: %v", err)
	}
	defer w.Close()

	var replayed []Entry
	err = w.ReplayCommitted(func(entry Entry) error {
		replayed = append(replayed, entry)
		return nil
	})
	if err != nil {
		t.Fatalf("ReplayCommitted failed: %v", err)
	}

	if len(replayed) != 1 {
		t.Fatalf("Expected 1 committed entry, got %d", len(replayed))
	}
	if replayed[0].Type != OpInsert {
		t.Fatalf("Expected OpInsert, got %v", replayed[0].Type)
	}
	if replayed[0].ID != 2 {
		t.Fatalf("Expected ID=2, got %d", replayed[0].ID)
	}
}

func TestWALCheckpoint(t *testing.T) {
	dir := t.TempDir()

	wal, err := New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()

	// Write some entries
	for i := uint64(1); i <= 5; i++ {
		err := wal.LogInsert(i, []float32{float32(i)}, encodeData(t, "data"), nil)
		if err != nil {
			t.Fatalf("LogInsert failed: %v", err)
		}
	}

	count, _ := wal.Len()
	// Each insert is written as Prepare+Commit.
	if count != 10 {
		t.Errorf("Expected 10 entries before checkpoint, got %d", count)
	}

	// Checkpoint
	err = wal.Checkpoint()
	if err != nil {
		t.Fatalf("Checkpoint failed: %v", err)
	}

	// Verify WAL is empty after checkpoint
	count, _ = wal.Len()
	if count != 0 {
		t.Errorf("Expected 0 entries after checkpoint, got %d", count)
	}

	// Add new entry after checkpoint
	err = wal.LogInsert(6, []float32{6.0}, encodeData(t, "data"), nil)
	if err != nil {
		t.Fatalf("LogInsert after checkpoint failed: %v", err)
	}

	count, _ = wal.Len()
	if count != 2 {
		t.Errorf("Expected 2 entries after checkpoint, got %d", count)
	}
}

func TestWALCorruptedFile(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "vecgo.wal")

	// Create a valid WAL first
	wal, err := New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}

	err = wal.LogInsert(1, []float32{1.0}, encodeData(t, "data"), nil)
	if err != nil {
		t.Fatalf("LogInsert failed: %v", err)
	}
	wal.Close()

	// Truncate file to corrupt it (remove last bytes)
	f, err := os.OpenFile(walPath, os.O_RDWR, 0644)
	if err != nil {
		t.Fatalf("Failed to open WAL: %v", err)
	}
	stat, _ := f.Stat()
	f.Truncate(stat.Size() - 10) // Remove last 10 bytes to corrupt entry
	f.Close()

	// Try to replay - should stop at corruption
	wal, err = New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to reopen WAL: %v", err)
	}
	defer wal.Close()

	replayed := 0
	err = wal.Replay(func(entry Entry) error {
		replayed++
		return nil
	})

	// Should have stopped before completing the corrupted entry
	// Binary format will fail to read the incomplete entry
	if replayed != 0 {
		t.Logf("Warning: Replayed %d entries (expected 0 due to truncation)", replayed)
	}
}

func TestWALSequenceNumbers(t *testing.T) {
	dir := t.TempDir()

	wal, err := New(func(o *Options) {
		o.Path = dir
	})
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}
	defer wal.Close()

	// Write entries and verify sequence numbers increase
	for i := uint64(1); i <= 3; i++ {
		err := wal.LogInsert(i, []float32{float32(i)}, encodeData(t, "data"), nil)
		if err != nil {
			t.Fatalf("LogInsert failed: %v", err)
		}
	}

	// Replay committed and verify sequence numbers
	replayed := []uint64{}
	err = wal.ReplayCommitted(func(entry Entry) error {
		replayed = append(replayed, entry.SeqNum)
		return nil
	})
	if err != nil {
		t.Fatalf("ReplayCommitted failed: %v", err)
	}

	if len(replayed) != 3 {
		t.Fatalf("Expected 3 committed ops, got %d", len(replayed))
	}

	// Each insert produces a Prepare then a Commit; ReplayCommitted uses commit seq nums (2,4,6,...).
	for i, seqNum := range replayed {
		expected := uint64((i + 1) * 2)
		if seqNum != expected {
			t.Errorf("Entry %d: expected seq %d, got %d", i, expected, seqNum)
		}
	}
}

func encodeData(t *testing.T, data string) []byte {
	b, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("Failed to encode data: %v", err)
	}
	return b
}

func TestWALCompression(t *testing.T) {
	dir := t.TempDir()

	// Create WAL with compression
	walCompressed, err := New(func(o *Options) {
		o.Path = filepath.Join(dir, "compressed")
		o.Compress = true
		o.CompressionLevel = 3
	})
	if err != nil {
		t.Fatalf("Failed to create compressed WAL: %v", err)
	}
	defer walCompressed.Close()

	// Create WAL without compression
	walUncompressed, err := New(func(o *Options) {
		o.Path = filepath.Join(dir, "uncompressed")
		o.Compress = false
	})
	if err != nil {
		t.Fatalf("Failed to create uncompressed WAL: %v", err)
	}
	defer walUncompressed.Close()

	// Insert same data to both
	const numEntries = 100
	for i := 0; i < numEntries; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		data := encodeData(t, "document-"+string(rune(i)))
		meta := metadata.Metadata{
			"id":   metadata.Int(int64(i)),
			"type": metadata.String("document"),
		}

		err := walCompressed.LogInsert(uint64(i), vector, data, meta)
		if err != nil {
			t.Fatalf("Compressed LogInsert failed: %v", err)
		}

		err = walUncompressed.LogInsert(uint64(i), vector, data, meta)
		if err != nil {
			t.Fatalf("Uncompressed LogInsert failed: %v", err)
		}
	}

	// Close to flush
	walCompressed.Close()
	walUncompressed.Close()

	// Compare file sizes
	compressedInfo, err := os.Stat(filepath.Join(dir, "compressed", "vecgo.wal"))
	if err != nil {
		t.Fatalf("Failed to stat compressed WAL: %v", err)
	}

	uncompressedInfo, err := os.Stat(filepath.Join(dir, "uncompressed", "vecgo.wal"))
	if err != nil {
		t.Fatalf("Failed to stat uncompressed WAL: %v", err)
	}

	compressionRatio := float64(uncompressedInfo.Size()) / float64(compressedInfo.Size())

	t.Logf("Compressed size:   %d bytes", compressedInfo.Size())
	t.Logf("Uncompressed size: %d bytes", uncompressedInfo.Size())
	t.Logf("Compression ratio: %.2fx", compressionRatio)

	// Verify compression is effective (should be at least 1.5x)
	if compressionRatio < 1.5 {
		t.Errorf("Compression ratio too low: %.2fx (expected >= 1.5x)", compressionRatio)
	}

	// Test replay with compression
	walCompressed2, err := New(func(o *Options) {
		o.Path = filepath.Join(dir, "compressed")
		o.Compress = true
	})
	if err != nil {
		t.Fatalf("Failed to reopen compressed WAL: %v", err)
	}
	defer walCompressed2.Close()

	// ReplayCommitted and verify
	entriesReplayed := 0
	err = walCompressed2.ReplayCommitted(func(entry Entry) error {
		entriesReplayed++
		return nil
	})
	if err != nil {
		t.Fatalf("ReplayCommitted failed: %v", err)
	}

	if entriesReplayed != numEntries {
		t.Errorf("Replayed %d entries, expected %d", entriesReplayed, numEntries)
	}

	t.Logf("Successfully replayed %d entries from compressed WAL", entriesReplayed)
}
