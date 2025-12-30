package metadata

import (
	"iter"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// Bitmap implements a 64-bit Roaring Bitmap.
// It wraps the official roaring64 implementation.
type Bitmap struct {
	rb *roaring64.Bitmap
}

// NewBitmap creates a new empty bitmap.
func NewBitmap() *Bitmap {
	return &Bitmap{
		rb: roaring64.New(),
	}
}

// Add adds an ID to the bitmap.
func (b *Bitmap) Add(id uint64) {
	b.rb.Add(id)
}

// Remove removes an ID from the bitmap.
func (b *Bitmap) Remove(id uint64) {
	b.rb.Remove(id)
}

// Contains checks if an ID is in the bitmap.
func (b *Bitmap) Contains(id uint64) bool {
	return b.rb.Contains(id)
}

// IsEmpty returns true if the bitmap is empty.
func (b *Bitmap) IsEmpty() bool {
	return b.rb.IsEmpty()
}

// Cardinality returns the number of elements in the bitmap.
func (b *Bitmap) Cardinality() uint64 {
	return b.rb.GetCardinality()
}

// Clone returns a deep copy of the bitmap.
func (b *Bitmap) Clone() *Bitmap {
	return &Bitmap{
		rb: b.rb.Clone(),
	}
}

// And computes the intersection of two bitmaps.
// Returns a new bitmap.
func (b *Bitmap) And(other *Bitmap) *Bitmap {
	return &Bitmap{
		rb: roaring64.And(b.rb, other.rb),
	}
}

// Or computes the union of two bitmaps.
// Returns a new bitmap.
func (b *Bitmap) Or(other *Bitmap) *Bitmap {
	return &Bitmap{
		rb: roaring64.Or(b.rb, other.rb),
	}
}

// Iterator returns an iterator over the IDs in the bitmap.
func (b *Bitmap) Iterator() iter.Seq[uint64] {
	return func(yield func(uint64) bool) {
		it := b.rb.Iterator()
		for it.HasNext() {
			if !yield(it.Next()) {
				return
			}
		}
	}
}

// ToArray returns the IDs as a slice.
func (b *Bitmap) ToArray() []uint64 {
	return b.rb.ToArray()
}

// GetSizeInBytes returns an estimate of the memory usage in bytes.
func (b *Bitmap) GetSizeInBytes() uint64 {
	return b.rb.GetSizeInBytes()
}
