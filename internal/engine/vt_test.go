package engine

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVersionedTombstones_Basic(t *testing.T) {
	vt := NewVersionedTombstones(10)

	// Not deleted initially
	assert.False(t, vt.IsDeleted(1, 100))

	// Mark deleted at LSN 5
	vt.MarkDeleted(1, 5)

	// Visibility
	assert.False(t, vt.IsDeleted(1, 4)) // Before delete
	assert.True(t, vt.IsDeleted(1, 5))  // At delete
	assert.True(t, vt.IsDeleted(1, 10)) // After delete

	// Mark deleted again at LSN 3 (older) -> should update to older
	// Wait, code says: "if current == 0 || current > lsn { update }"
	// In new code: "if currentLSN != 0 && currentLSN <= lsn { return }" (Already deleted earlier)
	// So if current is 5, and we insert 3. currentLSN <= lsn is 5 <= 3 (False). So we UPDATE.
	// Correct behavior: earlier deletion time should persist.

	vt.MarkDeleted(1, 3)
	assert.True(t, vt.IsDeleted(1, 3))
	assert.True(t, vt.IsDeleted(1, 4))
	assert.False(t, vt.IsDeleted(1, 2))
}

func TestVersionedTombstones_Grow(t *testing.T) {
	vt := NewVersionedTombstones(1)

	vt.MarkDeleted(100, 10)
	assert.True(t, vt.IsDeleted(100, 10))
	assert.False(t, vt.IsDeleted(100, 9))
}

func TestVersionedTombstones_Concurrency(t *testing.T) {
	vt := NewVersionedTombstones(10)
	var wg sync.WaitGroup

	// Writer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			vt.MarkDeleted(uint32(i), 10)
		}
	}()

	// Reader
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			// Just ensure no panic and consistent reads eventually
			_ = vt.IsDeleted(uint32(i), 11)
		}
	}()

	wg.Wait()

	for i := 0; i < 1000; i++ {
		assert.True(t, vt.IsDeleted(uint32(i), 10))
	}
}
