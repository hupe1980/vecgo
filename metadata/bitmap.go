package metadata

import (
	"iter"

	"github.com/RoaringBitmap/roaring/v2"
	"github.com/hupe1980/vecgo/core"
)

// LocalBitmap implements a 32-bit Roaring Bitmap.
// It wraps the official roaring implementation.
// Used for internal shard filtering (LocalID).
type LocalBitmap struct {
	rb *roaring.Bitmap
}

// NewLocalBitmap creates a new empty local bitmap.
func NewLocalBitmap() *LocalBitmap {
	return &LocalBitmap{
		rb: roaring.New(),
	}
}

// Add adds a LocalID to the bitmap.
func (b *LocalBitmap) Add(id core.LocalID) {
	b.rb.Add(uint32(id))
}

// Remove removes a LocalID from the bitmap.
func (b *LocalBitmap) Remove(id core.LocalID) {
	b.rb.Remove(uint32(id))
}

// Contains checks if a LocalID is in the bitmap.
func (b *LocalBitmap) Contains(id core.LocalID) bool {
	return b.rb.Contains(uint32(id))
}

// IsEmpty returns true if the bitmap is empty.
func (b *LocalBitmap) IsEmpty() bool {
	return b.rb.IsEmpty()
}

// Cardinality returns the number of elements in the bitmap.
func (b *LocalBitmap) Cardinality() uint64 {
	return b.rb.GetCardinality()
}

// Clone returns a deep copy of the bitmap.
func (b *LocalBitmap) Clone() *LocalBitmap {
	return &LocalBitmap{
		rb: b.rb.Clone(),
	}
}

// Iterator returns an iterator over the bitmap.
func (b *LocalBitmap) Iterator() iter.Seq[core.LocalID] {
	return func(yield func(core.LocalID) bool) {
		it := b.rb.Iterator()
		for it.HasNext() {
			if !yield(core.LocalID(it.Next())) {
				return
			}
		}
	}
}

// And computes the intersection of two bitmaps.
func (b *LocalBitmap) And(other *LocalBitmap) {
	b.rb.And(other.rb)
}

// Or computes the union of two bitmaps.
func (b *LocalBitmap) Or(other *LocalBitmap) {
	b.rb.Or(other.rb)
}

// Clear removes all elements from the bitmap.
func (b *LocalBitmap) Clear() {
	b.rb.Clear()
}

// GetSizeInBytes returns the size of the bitmap in bytes.
func (b *LocalBitmap) GetSizeInBytes() uint64 {
	return b.rb.GetSizeInBytes()
}
