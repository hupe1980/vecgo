package visited

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVisitedSet(t *testing.T) {
	v := New(10)

	// Test initial state
	assert.False(t, v.Visited(1))
	assert.False(t, v.Visited(5))

	// Test Visit
	v.Visit(1)
	assert.True(t, v.Visited(1))
	assert.False(t, v.Visited(5))

	v.Visit(5)
	assert.True(t, v.Visited(1))
	assert.True(t, v.Visited(5))

	// Test Reset
	v.Reset()
	assert.False(t, v.Visited(1))
	assert.False(t, v.Visited(5))

	// Test Visit after Reset
	v.Visit(1)
	assert.True(t, v.Visited(1))
	assert.False(t, v.Visited(5))

	// Test Resize
	v.Visit(15) // Should trigger resize
	assert.True(t, v.Visited(15))
	assert.True(t, v.Visited(1))
}

func TestVisitedSet_Resize(t *testing.T) {
	v := New(2)
	v.Visit(1)
	assert.True(t, v.Visited(1))

	v.Visit(5) // Should grow
	assert.True(t, v.Visited(5))
	assert.True(t, v.Visited(1))
}
