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
