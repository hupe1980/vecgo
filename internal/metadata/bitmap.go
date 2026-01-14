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
// The returned bitmap is owned by the caller and should NOT be returned to the pool.
// Use this for long-lived bitmaps (e.g., inverted index storage, API boundaries).
// For temporary bitmaps in hot paths, use GetPooledBitmap instead.
func NewLocalBitmap() *LocalBitmap {
	return &LocalBitmap{
		rb: roaring.New(),
	}
}

// GetPooledBitmap gets a bitmap from the pool for temporary use.
// IMPORTANT: Caller MUST call PutPooledBitmap when done to avoid pool exhaustion.
// The bitmap is cleared before being returned.
// Use this for temporary bitmaps in hot paths (e.g., filter evaluation).
// For owned bitmaps that outlive a function call, use NewLocalBitmap instead.
func GetPooledBitmap() *LocalBitmap {
	b := bitmapPool.Get().(*LocalBitmap)
	b.rb.Clear()
	return b
}

// PutPooledBitmap returns a pooled bitmap to the pool.
// Only call this for bitmaps obtained via GetPooledBitmap.
// Passing nil is safe and will be ignored.
// WARNING: Do not use the bitmap after calling this function.
func PutPooledBitmap(b *LocalBitmap) {
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
// The returned bitmap is owned (not pooled) and allocates a new roaring.Bitmap.
// For hot paths where you want to avoid allocation, use CloneTo with a pooled bitmap.
func (b *LocalBitmap) Clone() *LocalBitmap {
	return &LocalBitmap{
		rb: b.rb.Clone(),
	}
}

// CloneTo copies this bitmap's contents into the destination bitmap.
// The destination is cleared first, then filled via Or operation.
// This avoids roaring's internal Clone allocation when used with pooled bitmaps.
// Example:
//
//	dst := GetPooledBitmap()
//	src.CloneTo(dst)
//	// use dst...
//	PutPooledBitmap(dst)
func (b *LocalBitmap) CloneTo(dst *LocalBitmap) {
	dst.rb.Clear()
	dst.rb.Or(b.rb)
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
