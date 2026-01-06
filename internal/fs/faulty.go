package fs

import (
	"fmt"
	"os"
	"sync"
)

// FaultyFS is a FileSystem wrapper that can inject errors.
type FaultyFS struct {
	FS             FileSystem
	FailAfterBytes int64 // Fail writes after writing this many bytes
	written        int64
	mu             sync.Mutex
	Err            error
}

// GetWritten returns the total bytes written so far.
func (f *FaultyFS) GetWritten() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.written
}

// SetLimit sets the byte limit.
func (f *FaultyFS) SetLimit(limit int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.FailAfterBytes = limit
}

// NewFaultyFS creates a new FaultyFS wrapping the provided FS (or Default if nil).
func NewFaultyFS(fs FileSystem) *FaultyFS {
	if fs == nil {
		fs = Default
	}
	return &FaultyFS{
		FS:  fs,
		Err: fmt.Errorf("injected fault error"),
	}
}

func (f *FaultyFS) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	file, err := f.FS.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}
	// Wrap file to intercept writes
	return &faultyFile{File: file, fs: f}, nil
}

func (f *FaultyFS) Remove(name string) error {
	return f.FS.Remove(name)
}

func (f *FaultyFS) Rename(oldpath, newpath string) error {
	return f.FS.Rename(oldpath, newpath)
}

func (f *FaultyFS) Stat(name string) (os.FileInfo, error) {
	return f.FS.Stat(name)
}

func (f *FaultyFS) MkdirAll(path string, perm os.FileMode) error {
	return f.FS.MkdirAll(path, perm)
}

func (f *FaultyFS) ReadDir(name string) ([]os.DirEntry, error) {
	return f.FS.ReadDir(name)
}

func (f *FaultyFS) Truncate(name string, size int64) error {
	return f.FS.Truncate(name, size)
}

func (f *FaultyFS) checkWrite(n int) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.FailAfterBytes > 0 && f.written+int64(n) > f.FailAfterBytes {
		return f.Err
	}
	f.written += int64(n)
	return nil
}

type faultyFile struct {
	File
	fs *FaultyFS
}

func (ff *faultyFile) Write(p []byte) (n int, err error) {
	if err := ff.fs.checkWrite(len(p)); err != nil {
		// Simulate partial write? For now just fail.
		return 0, err
	}
	return ff.File.Write(p)
}
