package mmap

// Region represents a subsection of a memory mapping.
// It does not own the memory; the parent Mapping does.
type Region struct {
	parent *Mapping
	offset int
	size   int
}

// Region creates a new view into the mapping.
func (m *Mapping) Region(offset, size int) (*Region, error) {
	if m.closed.Load() {
		return nil, ErrClosed
	}
	if offset < 0 || size < 0 || offset+size > m.size {
		return nil, ErrOutOfBounds
	}
	return &Region{
		parent: m,
		offset: offset,
		size:   size,
	}, nil
}

// Bytes returns the byte slice for this region.
// Warning: The slice is valid only until the parent Mapping is closed.
func (r *Region) Bytes() []byte {
	// Check if parent is closed?
	// Accessing r.parent.data is safe if we check closed, but r.parent.data might be nil if closed.
	if r.parent.closed.Load() {
		return nil
	}
	return r.parent.data[r.offset : r.offset+r.size]
}

// Advise provides hints to the kernel about how this region will be accessed.
func (r *Region) Advise(pattern AccessPattern) error {
	if r.parent.closed.Load() {
		return ErrClosed
	}
	// We need to advise only the slice corresponding to this region.
	data := r.parent.data[r.offset : r.offset+r.size]
	return osAdvise(data, pattern)
}
