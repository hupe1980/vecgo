package manifest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/hupe1980/vecgo/blobstore"
	"github.com/hupe1980/vecgo/model"
)

const (
	ManifestFileName = "MANIFEST"
	CurrentFileName  = "CURRENT"
	// CurrentVersion is the version of the manifest format.
	// Version 1: Binary format with all fields
	CurrentVersion = 1
)

// Manifest describes the state of the engine at a specific point in time.
type Manifest struct {
	Version       int             `json:"version"`
	ID            uint64          `json:"id"`
	CreatedAt     time.Time       `json:"created_at"` // Timestamp of creation
	Dim           int             `json:"dim"`
	Metric        string          `json:"metric"`
	NextSegmentID model.SegmentID `json:"next_segment_id"`
	Segments      []SegmentInfo   `json:"segments"`
	PKIndex       PKIndexInfo     `json:"pk_index"`
	MaxLSN        uint64          `json:"max_lsn"`
}

// New creates a new empty manifest.
func New(dim int, metric string) *Manifest {
	return &Manifest{
		Version:       CurrentVersion,
		ID:            0,
		CreatedAt:     time.Now(),
		Dim:           dim,
		Metric:        metric,
		NextSegmentID: 1, // Start segment IDs at 1
	}
}

// SegmentInfo describes a single segment.
type SegmentInfo struct {
	ID       model.SegmentID `json:"id"`
	Level    int             `json:"level"`
	RowCount uint32          `json:"row_count"`
	Path     string          `json:"path"` // Relative to data dir
	Size     int64           `json:"size"` // Size in bytes
	// MinID and MaxID track the range of primary keys in this segment.
	// Used for calculating overlaps in leveled compaction.
	MinID model.ID `json:"min_id"`
	MaxID model.ID `json:"max_id"`
}

// PKIndexInfo describes the persistent PK index.
type PKIndexInfo struct {
	Path string `json:"path"` // Relative to data dir
}

// Store manages the manifest file and atomic updates.
type Store struct {
	store blobstore.BlobStore
	mu    sync.Mutex
}

// NewStore creates a new manifest store.
func NewStore(store blobstore.BlobStore) *Store {
	return &Store{store: store}
}

// Load loads the current manifest.
func (s *Store) Load(ctx context.Context) (*Manifest, error) {
	return s.LoadVersion(ctx, 0)
}

// LoadVersion loads a specific version ID. 0 means latest.
func (s *Store) LoadVersion(ctx context.Context, versionID uint64) (*Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var manifestFilename string
	if versionID == 0 {
		b, err := s.store.Open(ctx, CurrentFileName)
		if err != nil {
			// If CURRENT file is missing, we must map underlying store "Not Found" error
			// to manifest.ErrNotFound so engine can detect it and init new DB.
			// blobstore.ErrNotFound is os.ErrNotExist.
			if os.IsNotExist(err) || errors.Is(err, os.ErrNotExist) {
				return nil, ErrNotFound
			}
			return nil, err
		}
		defer b.Close()
		content, err := io.ReadAll(io.NewSectionReader(b, 0, b.Size()))
		if err != nil {
			return nil, err
		}
		manifestFilename = string(content)
	} else {
		manifestFilename = fmt.Sprintf("%s-%06d.bin", ManifestFileName, versionID)
	}

	b, err := s.store.Open(ctx, manifestFilename)
	if err != nil {
		return nil, fmt.Errorf("failed to open manifest %s: %w", manifestFilename, err)
	}
	defer b.Close()

	var m *Manifest
	if filepath.Ext(manifestFilename) == ".json" {
		m = &Manifest{}
		content, err := io.ReadAll(io.NewSectionReader(b, 0, b.Size()))
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(content, m); err != nil {
			return nil, err
		}
	} else {
		// Expect ReadBinary to be available (helper in binary.go)
		// Assuming signature: func ReadBinary(r io.Reader) (*Manifest, error)
		m, err = ReadBinary(io.NewSectionReader(b, 0, b.Size()))
		if err != nil {
			return nil, err
		}
	}

	return m, nil
}

// ListVersions returns all available manifest versions.
// Note: This method intentionally skips corrupted or unreadable manifests
// to provide a best-effort listing of available versions.
func (s *Store) ListVersions(ctx context.Context) ([]*Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	files, err := s.store.List(ctx, ManifestFileName)
	if err != nil {
		return nil, err
	}
	var manifests []*Manifest
	for _, f := range files {
		if filepath.Ext(f) != ".bin" && filepath.Ext(f) != ".json" {
			continue
		}
		b, err := s.store.Open(ctx, f)
		if err != nil {
			continue // Skip unreadable files
		}
		// In a real implementation we would only read the header (metrics, timestamp)
		// For now we read full for simplicity.
		var m *Manifest
		if filepath.Ext(f) == ".json" {
			m = &Manifest{}
			content, err := io.ReadAll(io.NewSectionReader(b, 0, b.Size()))
			if err != nil {
				b.Close()
				continue // Skip on read error
			}
			if err := json.Unmarshal(content, m); err != nil {
				b.Close()
				continue
			}
		} else {
			m, err = ReadBinary(io.NewSectionReader(b, 0, b.Size()))
			if err != nil {
				b.Close()
				continue // Skip corrupted binary manifests
			}
		}
		b.Close()
		if m != nil {
			manifests = append(manifests, m)
		}
	}
	return manifests, nil
}

// Save atomically saves a new manifest.
func (s *Store) Save(ctx context.Context, m *Manifest) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	m.Version = CurrentVersion
	m.ID++
	m.CreatedAt = time.Now()

	filename := fmt.Sprintf("%s-%06d.bin", ManifestFileName, m.ID)

	var buf bytes.Buffer
	if err := m.WriteBinary(&buf); err != nil {
		return err
	}

	// Atomic Write of Manifest Blob
	if err := s.store.Put(ctx, filename, buf.Bytes()); err != nil {
		return err
	}

	// Atomic Update of CURRENT
	// S3: Strong consistency on overwrites
	// Local: Atomic rename in Create/Put
	return s.store.Put(ctx, CurrentFileName, []byte(filename))
}

// DeleteVersion deletes the manifest file for the given version.
func (s *Store) DeleteVersion(ctx context.Context, versionID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Only binary format is written by Save(), so only delete .bin
	filename := fmt.Sprintf("%s-%06d.bin", ManifestFileName, versionID)
	return s.store.Delete(ctx, filename)
}
