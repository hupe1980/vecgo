package mmap

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMmap_Region_And_Advise(t *testing.T) {
	// Create temp file
	f, err := os.CreateTemp("", "mmaptest")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	size := 1024
	_, err = f.Write(make([]byte, size))
	require.NoError(t, err)
	f.Close()

	// Open mmap
	m, err := Open(f.Name())
	require.NoError(t, err)

	err = m.Advise(AccessRandom)
	require.NoError(t, err)

	// Region
	r, err := m.Region(100, 200)
	require.NoError(t, err)
	assert.Len(t, r.Bytes(), 200)

	err = r.Advise(AccessSequential)
	require.NoError(t, err)

	// Error cases
	_, err = m.Region(-1, 0)
	assert.Error(t, err)

	// Close parent
	err = m.Close()
	require.NoError(t, err)

	// Region after close
	assert.Nil(t, r.Bytes())
	assert.Error(t, r.Advise(AccessDefault))
}

func TestMmap_AfterClose(t *testing.T) {
	f, _ := os.CreateTemp("", "mmaptest2")
	defer os.Remove(f.Name())
	f.Write([]byte("data"))
	f.Close()

	m, _ := Open(f.Name())
	m.Close()

	// Methods after close
	assert.Nil(t, m.Bytes())
	assert.Error(t, m.Advise(AccessRandom))
	_, err := m.Region(0, 1)
	assert.Error(t, err)
}

func TestMapAnon(t *testing.T) {
	size := 4096

	m, err := MapAnon(size)
	require.NoError(t, err)
	require.NotNil(t, m)
	defer m.Close()

	assert.Equal(t, size, m.Size())

	// Anonymous mappings should be writable
	data := m.Bytes()
	require.Len(t, data, size)

	// Write and read back
	data[0] = 0xAB
	data[size-1] = 0xCD
	assert.Equal(t, byte(0xAB), data[0])
	assert.Equal(t, byte(0xCD), data[size-1])

	// Close should be idempotent
	err = m.Close()
	require.NoError(t, err)
	err = m.Close()
	require.NoError(t, err)

	// After close, Bytes returns nil
	assert.Nil(t, m.Bytes())
}

func TestMapAnon_InvalidSize(t *testing.T) {
	_, err := MapAnon(0)
	assert.Equal(t, ErrInvalidSize, err)

	_, err = MapAnon(-1)
	assert.Equal(t, ErrInvalidSize, err)
}

func TestMapping_ReadAt_AfterClose(t *testing.T) {
	f, _ := os.CreateTemp("", "mmaptest3")
	defer os.Remove(f.Name())
	f.Write([]byte("test data"))
	f.Close()

	m, _ := Open(f.Name())
	m.Close()

	buf := make([]byte, 4)
	_, err := m.ReadAt(buf, 0)
	assert.Equal(t, ErrClosed, err)
}

func TestRegion_OutOfBounds(t *testing.T) {
	f, _ := os.CreateTemp("", "mmaptest4")
	defer os.Remove(f.Name())
	f.Write(make([]byte, 100))
	f.Close()

	m, _ := Open(f.Name())
	defer m.Close()

	// Valid region
	r, err := m.Region(0, 100)
	require.NoError(t, err)
	assert.NotNil(t, r)

	// Out of bounds: offset + size > m.size
	_, err = m.Region(50, 100)
	assert.Equal(t, ErrOutOfBounds, err)

	// Out of bounds: negative offset
	_, err = m.Region(-1, 10)
	assert.Equal(t, ErrOutOfBounds, err)

	// Out of bounds: negative size
	_, err = m.Region(0, -1)
	assert.Equal(t, ErrOutOfBounds, err)
}
