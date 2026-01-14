package imetadata

import (
	"io"
	"iter"
	"sync"

	"github.com/RoaringBitmap/roaring/v2"
	"github.com/hupe1980/vecgo/model"
)

// LocalBitmap implements a 32-bit Roaring Bitmap.
// It wraps the official roaring implementation.
// Used for internal row filtering (RowID).
type LocalBitmap struct {
	rb *roaring.Bitmap
}

// bitmapPool is a sync.Pool for reusing LocalBitmap instances.
// This reduces allocations in filtered search hot paths.
var bitmapPool = sync.Pool{
	New: func() any {
		return &LocalBitmap{
			rb: roaring.New(),
		}
	},
}

// NewLocalBitmap creates a new empty local bitmap.
func NewLocalBitmap() *LocalBitmap {
	return &LocalBitmap{
		rb: roaring.New(),
	}
}

// GetBitmap gets a bitmap from the pool. Call PutBitmap when done.
func GetBitmap() *LocalBitmap {
	b := bitmapPool.Get().(*LocalBitmap)
	b.rb.Clear()
	return b
}

// PutBitmap returns a bitmap to the pool.
func PutBitmap(b *LocalBitmap) {
	if b == nil {
		return
	}
	// Clear before returning to pool to release container memory
	b.rb.Clear()
	bitmapPool.Put(b)
}

// Add adds a RowID to the bitmap.
func (b *LocalBitmap) Add(id uint32) {
	b.rb.Add(id)
}

// Remove removes a RowID from the bitmap.
func (b *LocalBitmap) Remove(id uint32) {
	b.rb.Remove(id)
}

// Contains checks if a RowID is in the bitmap.
func (b *LocalBitmap) Contains(id uint32) bool {
	return b.rb.Contains(id)
}

// ForEach iterates over the bitmap.
func (b *LocalBitmap) ForEach(fn func(id uint32) bool) {
	it := b.rb.Iterator()
	for it.HasNext() {
		if !fn(it.Next()) {
			break
		}
	}
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
func (b *LocalBitmap) Iterator() iter.Seq[model.RowID] {
	return func(yield func(model.RowID) bool) {
		it := b.rb.Iterator()
		for it.HasNext() {
			if !yield(model.RowID(it.Next())) {
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

// WriteTo writes the bitmap to an io.Writer.
func (b *LocalBitmap) WriteTo(w io.Writer) (int64, error) {
	return b.rb.WriteTo(w)
}

// ReadFrom reads the bitmap from an io.Reader.
func (b *LocalBitmap) ReadFrom(r io.Reader) (int64, error) {
	return b.rb.ReadFrom(r)
}
