package metadata

import (
	"iter"

	"github.com/RoaringBitmap/roaring/v2"
)

// Bitmap implements a 32-bit Roaring Bitmap.
// It wraps the official roaring implementation.
// Note: We use 32-bit bitmaps because LocalIDs are uint32.
type Bitmap struct {
	rb *roaring.Bitmap
}

// NewBitmap creates a new empty bitmap.
func NewBitmap() *Bitmap {
	return &Bitmap{
		rb: roaring.New(),
	}
}

// Add adds an ID to the bitmap.
func (b *Bitmap) Add(id uint64) {
	b.rb.Add(uint32(id))
}

// Remove removes an ID from the bitmap.
func (b *Bitmap) Remove(id uint64) {
	b.rb.Remove(uint32(id))
}

// Contains checks if an ID is in the bitmap.
func (b *Bitmap) Contains(id uint64) bool {
	return b.rb.Contains(uint32(id))
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
		rb: roaring.And(b.rb, other.rb),
	}
}

// Or computes the union of two bitmaps.
// Returns a new bitmap.
func (b *Bitmap) Or(other *Bitmap) *Bitmap {
	return &Bitmap{
		rb: roaring.Or(b.rb, other.rb),
	}
}

// OrInPlace computes the union of two bitmaps in place.
func (b *Bitmap) OrInPlace(other *Bitmap) {
	b.rb.Or(other.rb)
}

// AndInPlace computes the intersection of two bitmaps in place.
func (b *Bitmap) AndInPlace(other *Bitmap) {
	b.rb.And(other.rb)
}

// Clear clears the bitmap.
func (b *Bitmap) Clear() {
	b.rb.Clear()
}

// Iterator returns an iterator over the IDs in the bitmap.
func (b *Bitmap) Iterator() iter.Seq[uint64] {
	return func(yield func(uint64) bool) {
		it := b.rb.Iterator()
		for it.HasNext() {
			if !yield(uint64(it.Next())) {
				return
			}
		}
	}
}

// ToArray returns the IDs as a slice.
func (b *Bitmap) ToArray() []uint64 {
	arr := b.rb.ToArray()
	res := make([]uint64, len(arr))
	for i, v := range arr {
		res[i] = uint64(v)
	}
	return res
}

// GetSizeInBytes returns an estimate of the memory usage in bytes.
func (b *Bitmap) GetSizeInBytes() uint64 {
	return b.rb.GetSizeInBytes()
}
