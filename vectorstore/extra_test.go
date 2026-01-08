package vectorstore

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFormat_Helpers(t *testing.T) {
	h := FileHeader{
		Count:     10,
		Dimension: 128,
		Flags:     FlagHasVersions,
	}
	assert.True(t, h.HasVersions())
	assert.Equal(t, int64(10*8), h.VersionDataSize())

	// Check calc: Header + vec + bitmap + version + checksum
	// vec: 10*128*4 = 5120
	// bitmap: ceil(10/8) = 2
	// version: 10*8 = 80
	// HeaderSize = 12 (likely? Need to check const or inferred)
	// Checksum = 4

	// Just verify TotalSize combines them
	expected := int64(HeaderSize) + h.VectorDataSize() + h.BitmapSize() + h.VersionDataSize() + 4
	assert.Equal(t, expected, h.TotalSize())

	h2 := FileHeader{Flags: 0, Count: 10, Dimension: 128}
	assert.False(t, h2.HasVersions())
	assert.Equal(t, int64(0), h2.VersionDataSize())
}

func TestMmap_Extras(t *testing.T) {
	// Setup Mmap file
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "mmap_extra.col")

	s, _ := New(2)
	s.Append([]float32{1.0, 1.0})
	s.Append([]float32{2.0, 2.0})
	s.DeleteVector(0)

	f, err := os.Create(filename)
	require.NoError(t, err)
	_, err = s.WriteTo(f)
	require.NoError(t, err)
	f.Close()

	ms, closer, err := OpenMmap(filename)
	require.NoError(t, err)
	defer closer.Close()

	// IsDeleted
	assert.True(t, ms.IsDeleted(0))
	assert.False(t, ms.IsDeleted(1))

	// Iterate
	count := 0
	ms.Iterate(func(id model.RowID, vec []float32) bool {
		count++
		assert.Equal(t, model.RowID(1), id)
		assert.Equal(t, float32(2.0), vec[0])
		return true
	})
	assert.Equal(t, 1, count)

	// RawData
	data := ms.RawData()
	// Total floats for 2 vectors of dim 2: 2 * 2 = 4 floats
	assert.Equal(t, 4, len(data))

	// GetVectorUnsafe check
	v, ok := ms.GetVectorUnsafe(1)
	assert.True(t, ok)
	assert.Equal(t, float32(2.0), v[0])

	_, ok = ms.GetVectorUnsafe(0) // deleted but unsafe returns it
	assert.True(t, ok)

	_, ok = ms.GetVectorUnsafe(999) // outbound
	assert.False(t, ok)
}

func TestColumnar_Extras(t *testing.T) {
	s, _ := New(2)
	s.Append([]float32{1.0, 1.0})

	// Size
	assert.Greater(t, s.Size(), int64(0))

	// GetVectorUnsafe
	v, ok := s.GetVectorUnsafe(0)
	assert.True(t, ok)
	assert.Equal(t, float32(1.0), v[0])

	_, ok = s.GetVectorUnsafe(99)
	assert.False(t, ok)

	// Close (noop likely)
	assert.NoError(t, s.Close())
}
