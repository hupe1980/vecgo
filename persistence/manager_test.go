package persistence

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/wal"
)

// mockSnapshotable implements Snapshotable for testing.
type mockSnapshotable struct {
	data []byte
}

func (m *mockSnapshotable) Save(_ context.Context, w io.Writer) error {
	_, err := w.Write(m.data)
	return err
}

// mockSnapshotLoader implements SnapshotLoader for testing.
type mockSnapshotLoader struct {
	loadedPath string
	loadedData []byte
}

func (m *mockSnapshotLoader) LoadSnapshot(_ context.Context, path string) error {
	m.loadedPath = path
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	m.loadedData = data
	return nil
}

// mockWALReplayer implements WALReplayer for testing.
type mockWALReplayer struct {
	replayedEntries []wal.Entry
}

func (m *mockWALReplayer) ReplayEntry(_ context.Context, entry wal.Entry) error {
	m.replayedEntries = append(m.replayedEntries, entry)
	return nil
}

func TestNewManager(t *testing.T) {
	t.Run("with defaults", func(t *testing.T) {
		pm, err := NewManager(ManagerOptions{})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		if pm.WAL() != nil {
			t.Error("Expected WAL to be nil without WALPath")
		}
		if pm.SnapshotPath() != "" {
			t.Error("Expected empty snapshot path")
		}
		if pm.Codec() == nil {
			t.Error("Expected default codec to be set")
		}
	})

	t.Run("with WAL", func(t *testing.T) {
		tmpDir := t.TempDir()

		pm, err := NewManager(ManagerOptions{
			WALPath: tmpDir,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		if pm.WAL() == nil {
			t.Error("Expected WAL to be initialized")
		}
	})

	t.Run("with snapshot path", func(t *testing.T) {
		tmpDir := t.TempDir()
		snapshotPath := filepath.Join(tmpDir, "test.snap")

		pm, err := NewManager(ManagerOptions{
			SnapshotPath: snapshotPath,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		if pm.SnapshotPath() != snapshotPath {
			t.Errorf("SnapshotPath() = %v, want %v", pm.SnapshotPath(), snapshotPath)
		}
	})
}

func TestManagerSnapshot(t *testing.T) {
	t.Run("atomic snapshot", func(t *testing.T) {
		tmpDir := t.TempDir()
		snapshotPath := filepath.Join(tmpDir, "test.snap")

		pm, err := NewManager(ManagerOptions{
			SnapshotPath: snapshotPath,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		testData := []byte("test snapshot data")
		ctx := context.Background()
		err = pm.Snapshot(ctx, func(_ context.Context, w io.Writer) error {
			_, err := w.Write(testData)
			return err
		})
		if err != nil {
			t.Fatalf("Snapshot() error = %v", err)
		}

		// Verify snapshot was written
		data, err := os.ReadFile(snapshotPath)
		if err != nil {
			t.Fatalf("Failed to read snapshot: %v", err)
		}
		if !bytes.Equal(data, testData) {
			t.Errorf("Snapshot data = %v, want %v", data, testData)
		}
	})

	t.Run("snapshot without path", func(t *testing.T) {
		pm, err := NewManager(ManagerOptions{})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		ctx := context.Background()
		err = pm.Snapshot(ctx, func(_ context.Context, w io.Writer) error {
			return nil
		})
		if err != ErrNoSnapshotPath {
			t.Errorf("Snapshot() error = %v, want %v", err, ErrNoSnapshotPath)
		}
	})

	t.Run("snapshot to path", func(t *testing.T) {
		tmpDir := t.TempDir()
		customPath := filepath.Join(tmpDir, "custom.snap")

		pm, err := NewManager(ManagerOptions{})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		testData := []byte("custom path data")
		ctx := context.Background()
		err = pm.SnapshotToPath(ctx, customPath, func(_ context.Context, w io.Writer) error {
			_, err := w.Write(testData)
			return err
		})
		if err != nil {
			t.Fatalf("SnapshotToPath() error = %v", err)
		}

		// Verify snapshot was written
		data, err := os.ReadFile(customPath)
		if err != nil {
			t.Fatalf("Failed to read snapshot: %v", err)
		}
		if !bytes.Equal(data, testData) {
			t.Errorf("Snapshot data = %v, want %v", data, testData)
		}
	})

	t.Run("snapshot respects cancellation", func(t *testing.T) {
		tmpDir := t.TempDir()
		snapshotPath := filepath.Join(tmpDir, "test.snap")

		pm, err := NewManager(ManagerOptions{
			SnapshotPath: snapshotPath,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		err = pm.Snapshot(ctx, func(_ context.Context, w io.Writer) error {
			return nil
		})
		if err != context.Canceled {
			t.Errorf("Snapshot() error = %v, want %v", err, context.Canceled)
		}
	})
}

func TestManagerRecover(t *testing.T) {
	t.Run("recover from snapshot", func(t *testing.T) {
		tmpDir := t.TempDir()
		snapshotPath := filepath.Join(tmpDir, "test.snap")

		// Create snapshot file
		testData := []byte("recovery test data")
		if err := os.WriteFile(snapshotPath, testData, 0644); err != nil {
			t.Fatalf("Failed to write test snapshot: %v", err)
		}

		pm, err := NewManager(ManagerOptions{
			SnapshotPath: snapshotPath,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		loader := &mockSnapshotLoader{}
		replayer := &mockWALReplayer{}

		ctx := context.Background()
		err = pm.Recover(ctx, loader, replayer)
		if err != nil {
			t.Fatalf("Recover() error = %v", err)
		}

		if loader.loadedPath != snapshotPath {
			t.Errorf("Loaded path = %v, want %v", loader.loadedPath, snapshotPath)
		}
		if !bytes.Equal(loader.loadedData, testData) {
			t.Errorf("Loaded data = %v, want %v", loader.loadedData, testData)
		}
	})

	t.Run("recover without snapshot", func(t *testing.T) {
		tmpDir := t.TempDir()
		snapshotPath := filepath.Join(tmpDir, "nonexistent.snap")

		pm, err := NewManager(ManagerOptions{
			SnapshotPath: snapshotPath,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		loader := &mockSnapshotLoader{}
		replayer := &mockWALReplayer{}

		// Should succeed even without snapshot
		ctx := context.Background()
		err = pm.Recover(ctx, loader, replayer)
		if err != nil {
			t.Fatalf("Recover() error = %v", err)
		}

		if loader.loadedPath != "" {
			t.Errorf("Expected no snapshot to be loaded, got path: %v", loader.loadedPath)
		}
	})

	t.Run("recover respects cancellation", func(t *testing.T) {
		tmpDir := t.TempDir()
		snapshotPath := filepath.Join(tmpDir, "test.snap")

		// Create snapshot file
		if err := os.WriteFile(snapshotPath, []byte("data"), 0644); err != nil {
			t.Fatalf("Failed to write test snapshot: %v", err)
		}

		pm, err := NewManager(ManagerOptions{
			SnapshotPath: snapshotPath,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		loader := &mockSnapshotLoader{}
		replayer := &mockWALReplayer{}

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		err = pm.Recover(ctx, loader, replayer)
		if err != context.Canceled {
			t.Errorf("Recover() error = %v, want %v", err, context.Canceled)
		}
	})
}

func TestManagerClose(t *testing.T) {
	t.Run("close without WAL", func(t *testing.T) {
		pm, err := NewManager(ManagerOptions{})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}

		if err := pm.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		// Operations should fail after close
		ctx := context.Background()
		err = pm.Snapshot(ctx, func(_ context.Context, w io.Writer) error { return nil })
		if err != ErrManagerClosed {
			t.Errorf("Snapshot() after close error = %v, want %v", err, ErrManagerClosed)
		}
	})

	t.Run("close with WAL", func(t *testing.T) {
		tmpDir := t.TempDir()

		pm, err := NewManager(ManagerOptions{
			WALPath: tmpDir,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}

		if err := pm.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		// Double close should be safe
		if err := pm.Close(); err != nil {
			t.Fatalf("Double Close() error = %v", err)
		}
	})
}

func TestManagerCheckpoint(t *testing.T) {
	t.Run("checkpoint without WAL", func(t *testing.T) {
		pm, err := NewManager(ManagerOptions{})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		err = pm.Checkpoint()
		if err != ErrNoWAL {
			t.Errorf("Checkpoint() error = %v, want %v", err, ErrNoWAL)
		}
	})

	t.Run("checkpoint with WAL", func(t *testing.T) {
		tmpDir := t.TempDir()

		pm, err := NewManager(ManagerOptions{
			WALPath: tmpDir,
		})
		if err != nil {
			t.Fatalf("NewManager() error = %v", err)
		}
		defer pm.Close()

		// Checkpoint should succeed
		if err := pm.Checkpoint(); err != nil {
			t.Errorf("Checkpoint() error = %v", err)
		}
	})
}

func TestAtomicSaveToDir(t *testing.T) {
	t.Run("save multiple files", func(t *testing.T) {
		tmpDir := t.TempDir()

		files := map[string]func(io.Writer) error{
			"file1.bin": func(w io.Writer) error {
				_, err := w.Write([]byte("file1 content"))
				return err
			},
			"file2.bin": func(w io.Writer) error {
				_, err := w.Write([]byte("file2 content"))
				return err
			},
		}

		err := AtomicSaveToDir(tmpDir, files)
		if err != nil {
			t.Fatalf("AtomicSaveToDir() error = %v", err)
		}

		// Verify files were written
		data1, err := os.ReadFile(filepath.Join(tmpDir, "file1.bin"))
		if err != nil {
			t.Fatalf("Failed to read file1: %v", err)
		}
		if string(data1) != "file1 content" {
			t.Errorf("file1 content = %q, want %q", data1, "file1 content")
		}

		data2, err := os.ReadFile(filepath.Join(tmpDir, "file2.bin"))
		if err != nil {
			t.Fatalf("Failed to read file2: %v", err)
		}
		if string(data2) != "file2 content" {
			t.Errorf("file2 content = %q, want %q", data2, "file2 content")
		}
	})

	t.Run("creates directory if needed", func(t *testing.T) {
		tmpDir := t.TempDir()
		subDir := filepath.Join(tmpDir, "subdir", "nested")

		files := map[string]func(io.Writer) error{
			"test.bin": func(w io.Writer) error {
				_, err := w.Write([]byte("test"))
				return err
			},
		}

		err := AtomicSaveToDir(subDir, files)
		if err != nil {
			t.Fatalf("AtomicSaveToDir() error = %v", err)
		}

		// Verify directory and file were created
		if _, err := os.Stat(filepath.Join(subDir, "test.bin")); err != nil {
			t.Errorf("File not created: %v", err)
		}
	})

	t.Run("no temp files left on error", func(t *testing.T) {
		tmpDir := t.TempDir()

		files := map[string]func(io.Writer) error{
			"file1.bin": func(w io.Writer) error {
				_, err := w.Write([]byte("file1 content"))
				return err
			},
			"file2.bin": func(w io.Writer) error {
				return io.ErrShortWrite // Simulate error
			},
		}

		err := AtomicSaveToDir(tmpDir, files)
		if err == nil {
			t.Fatal("Expected error, got nil")
		}

		// Check for temp files
		entries, _ := os.ReadDir(tmpDir)
		for _, entry := range entries {
			if filepath.Ext(entry.Name()) == "" && len(entry.Name()) > 10 {
				t.Errorf("Temp file not cleaned up: %s", entry.Name())
			}
		}
	})
}

func TestManagerConcurrency(t *testing.T) {
	tmpDir := t.TempDir()
	snapshotPath := filepath.Join(tmpDir, "concurrent.snap")

	pm, err := NewManager(ManagerOptions{
		SnapshotPath: snapshotPath,
	})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}
	defer pm.Close()

	var wg sync.WaitGroup
	const numGoroutines = 10

	ctx := context.Background()

	// Concurrent snapshots
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			data := []byte("snapshot data " + string(rune('0'+n)))
			_ = pm.Snapshot(ctx, func(_ context.Context, w io.Writer) error {
				_, err := w.Write(data)
				return err
			})
		}(i)
	}

	wg.Wait()

	// Verify final snapshot exists and is valid
	if _, err := os.Stat(snapshotPath); err != nil {
		t.Errorf("Snapshot not found after concurrent writes: %v", err)
	}
}

func TestManagerSetSnapshotPath(t *testing.T) {
	pm, err := NewManager(ManagerOptions{})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}
	defer pm.Close()

	if pm.SnapshotPath() != "" {
		t.Error("Expected empty initial snapshot path")
	}

	tmpDir := t.TempDir()
	newPath := filepath.Join(tmpDir, "new.snap")
	pm.SetSnapshotPath(newPath)

	if pm.SnapshotPath() != newPath {
		t.Errorf("SnapshotPath() = %v, want %v", pm.SnapshotPath(), newPath)
	}

	// Verify we can now snapshot
	ctx := context.Background()
	err = pm.Snapshot(ctx, func(_ context.Context, w io.Writer) error {
		_, err := w.Write([]byte("test"))
		return err
	})
	if err != nil {
		t.Errorf("Snapshot() error after SetSnapshotPath = %v", err)
	}
}

func TestNewManagerWithWAL(t *testing.T) {
	tmpDir := t.TempDir()

	// Create WAL first
	w, err := wal.New(func(o *wal.Options) {
		o.Path = tmpDir
	})
	if err != nil {
		t.Fatalf("wal.New() error = %v", err)
	}
	defer w.Close()

	snapshotPath := filepath.Join(tmpDir, "test.snap")
	pm := NewManagerWithWAL(snapshotPath, w, nil)

	if pm.WAL() != w {
		t.Error("Expected WAL to match provided instance")
	}
	if pm.SnapshotPath() != snapshotPath {
		t.Errorf("SnapshotPath() = %v, want %v", pm.SnapshotPath(), snapshotPath)
	}

	// Close manager (should not close the WAL since we didn't create it)
	if err := pm.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}
}
