package zerocopy

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStore_Basic(t *testing.T) {
	// Initialize store
	store := New(3) // dim=3

	// Pre-allocate storage for 2 vectors (ids 0 and 1)
	// zerocopy.Store requires pre-allocated backing storage
	data := make([]float32, 2*3)
	store.SetData(data)

	// Set vector
	vec1 := []float32{1.0, 2.0, 3.0}
	err := store.SetVector(1, vec1)
	require.NoError(t, err)

	// Get vector
	vec, found := store.GetVector(1)
	assert.True(t, found)
	assert.Equal(t, vec1, vec)

	// Update vector
	vec2 := []float32{4.0, 5.0, 6.0}
	err = store.SetVector(1, vec2)
	require.NoError(t, err)

	vec, found = store.GetVector(1)
	assert.True(t, found)
	assert.Equal(t, vec2, vec)

	// Delete vector (no-op in zerocopy, but shouldn't error)
	err = store.DeleteVector(1)
	require.NoError(t, err)

	// Verify vector is still there (zerocopy doesn't support true deletion)
	// or maybe it zeroes it out? The implementation currently does nothing.
	// Let's check the implementation of DeleteVector.
	// It returns nil. So the vector should still be retrievable.
	vec, found = store.GetVector(1)
	assert.True(t, found)
	assert.Equal(t, vec2, vec)
}

func TestStore_DimensionMismatch(t *testing.T) {
	store := New(3)

	err := store.SetVector(1, []float32{1.0, 2.0}) // dim 2
	assert.Error(t, err)
}
