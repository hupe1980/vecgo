package vecgo

// Close releases resources held by this Vecgo instance.
//
// This is primarily useful for mmap-backed loads (vecgo_mmap build tag), but it
// also closes WAL if it is enabled.
func (vg *Vecgo[T]) Close() error {
	if vg == nil {
		return nil
	}
	var firstErr error
	if vg.wal != nil {
		if err := vg.wal.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if vg.mmapCloser != nil {
		if err := vg.mmapCloser.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		vg.mmapCloser = nil
	}
	return firstErr
}
