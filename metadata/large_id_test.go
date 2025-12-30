package metadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUnifiedIndex_LargeID(t *testing.T) {
	ui := NewUnifiedIndex()
	var largeID uint64 = 1 << 33 // > 2^32

	doc := Document{
		"tag": String("test"),
	}

	ui.Set(largeID, doc)

	retrieved, ok := ui.Get(largeID)
	assert.True(t, ok)
	assert.Equal(t, doc, retrieved)
}
