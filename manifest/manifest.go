package manifest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/hupe1980/vecgo/model"
)

const (
	ManifestFileName = "MANIFEST"
	CurrentFileName  = "CURRENT"
)

// Manifest describes the state of the engine at a specific point in time.
type Manifest struct {
	ID            uint64          `json:"id"`
	NextSegmentID model.SegmentID `json:"next_segment_id"`
	Segments      []SegmentInfo   `json:"segments"`
	PKIndex       PKIndexInfo     `json:"pk_index"`
	MaxLSN        uint64          `json:"max_lsn"`
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
	dir string
	mu  sync.Mutex
}

// NewStore creates a new manifest store.
func NewStore(dir string) *Store {
	return &Store{dir: dir}
}

// Load loads the current manifest.
func (s *Store) Load() (*Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Read CURRENT file to get the manifest filename
	currentPath := filepath.Join(s.dir, CurrentFileName)
	content, err := os.ReadFile(currentPath)
	if os.IsNotExist(err) {
		// No manifest yet, return empty
		return &Manifest{ID: 0}, nil
	}
	if err != nil {
		return nil, err
	}

	manifestPath := filepath.Join(s.dir, string(content))
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	return &m, nil
}

// Save atomically saves a new manifest.
func (s *Store) Save(m *Manifest) error {
	s.mu.Lock()
	defer s.mu.Unlock()

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
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	// Rename to final manifest file
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}

	// Sync directory to persist rename
	if err := syncDir(s.dir); err != nil {
		return err
	}

	// 2. Update CURRENT pointer atomically
	currentTmp := filepath.Join(s.dir, CurrentFileName+".tmp")
	cf, err := os.Create(currentTmp)
	if err != nil {
		return err
	}
	if _, err := cf.WriteString(filename); err != nil {
		cf.Close()
		return err
	}
	if err := cf.Sync(); err != nil {
		cf.Close()
		return err
	}
	if err := cf.Close(); err != nil {
		return err
	}

	if err := os.Rename(currentTmp, filepath.Join(s.dir, CurrentFileName)); err != nil {
		return err
	}

	return syncDir(s.dir)
}

func syncDir(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
