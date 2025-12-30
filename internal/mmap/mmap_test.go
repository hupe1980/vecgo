package mmap

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMmap_OpenReadClose(t *testing.T) {
	// Create a file with some data
	content := []byte("Hello, Mmap!")
	f, err := os.CreateTemp("", "mmap_test")
	require.NoError(t, err)
	defer os.Remove(f.Name())

	_, err = f.Write(content)
	require.NoError(t, err)
	f.Close()

	// Open mmap
	m, err := Open(f.Name())
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, int64(len(content)), int64(len(m.Data)))

	// ReadAt
	buf := make([]byte, 5)
	n, err := m.ReadAt(buf, 7) // "Mmap!"
	require.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, "Mmap!", string(buf))

	// ReadAt out of bounds
	buf2 := make([]byte, 10)
	_, err = m.ReadAt(buf2, 100)
	assert.Error(t, err)
}

func TestMmap_EmptyFile(t *testing.T) {
	f, err := os.CreateTemp("", "mmap_test_empty")
	require.NoError(t, err)
	defer os.Remove(f.Name())
	f.Close()

	m, err := Open(f.Name())
	require.NoError(t, err)
	defer m.Close()

	assert.Equal(t, int64(0), int64(len(m.Data)))
}
