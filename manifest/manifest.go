package manifest

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/hupe1980/vecgo/internal/fs"
	"github.com/hupe1980/vecgo/model"
)

const (
	ManifestFileName = "MANIFEST"
	CurrentFileName  = "CURRENT"
	CurrentVersion   = 1
)

// Manifest describes the state of the engine at a specific point in time.
type Manifest struct {
	Version       int             `json:"version"`
	ID            uint64          `json:"id"`
	NextSegmentID model.SegmentID `json:"next_segment_id"`
	Segments      []SegmentInfo   `json:"segments"`
	PKIndex       PKIndexInfo     `json:"pk_index"`
	MaxLSN        uint64          `json:"max_lsn"`
	WALID         uint64          `json:"wal_id"` // ID of the current WAL file
}

// SegmentInfo describes a single segment.
type SegmentInfo struct {
	ID       model.SegmentID `json:"id"`
	Level    int             `json:"level"`
	RowCount uint32          `json:"row_count"`
	Path     string          `json:"path"` // Relative to data dir
}

// PKIndexInfo describes the persistent PK index.
type PKIndexInfo struct {
	Path string `json:"path"` // Relative to data dir
}

// Store manages the manifest file and atomic updates.
type Store struct {
	fs  fs.FileSystem
	dir string
	mu  sync.Mutex
}

// NewStore creates a new manifest store.
func NewStore(fsys fs.FileSystem, dir string) *Store {
	return &Store{
		fs:  fsys,
		dir: dir,
	}
}

// Load loads the current manifest.
func (s *Store) Load() (*Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	readFile := func(path string) ([]byte, error) {
		f, err := s.fs.OpenFile(path, os.O_RDONLY, 0)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return io.ReadAll(f)
	}

	// Read CURRENT file to get the manifest filename
	currentPath := filepath.Join(s.dir, CurrentFileName)
	content, err := readFile(currentPath)
	if os.IsNotExist(err) {
		// No manifest yet, return empty with current version
		return &Manifest{ID: 0, Version: CurrentVersion}, nil
	}
	if err != nil {
		return nil, err
	}

	manifestPath := filepath.Join(s.dir, string(content))
	data, err := readFile(manifestPath)
	if err != nil {
		return nil, err
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	if m.Version != CurrentVersion {
		return nil, fmt.Errorf("unsupported manifest version: %d (expected %d)", m.Version, CurrentVersion)
	}

	return &m, nil
}

// Save atomically saves a new manifest.
func (s *Store) Save(m *Manifest) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	m.Version = CurrentVersion
	m.ID++

	// 1. Write new manifest file
	filename := fmt.Sprintf("%s-%06d.json", ManifestFileName, m.ID)
	path := filepath.Join(s.dir, filename)

	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}

	// Write to temp file first
	tmpPath := path + ".tmp"
	f, err := s.fs.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		s.fs.Remove(tmpPath)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		s.fs.Remove(tmpPath)
		return err
	}
	if err := f.Close(); err != nil {
		s.fs.Remove(tmpPath)
		return err
	}

	// Rename to final manifest file
	if err := s.fs.Rename(tmpPath, path); err != nil {
		s.fs.Remove(tmpPath)
		return err
	}

	// Sync directory to persist rename
	if err := s.syncDir(s.dir); err != nil {
		return err
	}

	// 2. Update CURRENT pointer atomically
	currentTmp := filepath.Join(s.dir, CurrentFileName+".tmp")
	cf, err := s.fs.OpenFile(currentTmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	if _, err := cf.Write([]byte(filename)); err != nil {
		cf.Close()
		s.fs.Remove(currentTmp)
		return err
	}
	if err := cf.Sync(); err != nil {
		cf.Close()
		s.fs.Remove(currentTmp)
		return err
	}
	if err := cf.Close(); err != nil {
		s.fs.Remove(currentTmp)
		return err
	}

	if err := s.fs.Rename(currentTmp, filepath.Join(s.dir, CurrentFileName)); err != nil {
		s.fs.Remove(currentTmp)
		return err
	}

	return s.syncDir(s.dir)
}

func (s *Store) syncDir(dir string) error {
	f, err := s.fs.OpenFile(dir, os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
