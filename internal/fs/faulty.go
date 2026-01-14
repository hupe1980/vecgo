package fs

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

// Fault defines specific failure behavior for a file.
type Fault struct {
	// FailAfterBytes fails writes after this many bytes written to this file.
	// -1 disables the per-file limit.
	FailAfterBytes int64
	// FailOnSync causes Sync() to return an error.
	FailOnSync bool
	// FailOnClose causes Close() to return an error (file is still closed).
	FailOnClose bool
	// Err is the error to return. If nil, a default error is used.
	Err error
}

// FaultyFS is a FileSystem wrapper that injects errors for testing.
// It supports both global write limits and per-file fault rules.
type FaultyFS struct {
	FS FileSystem

	mu          sync.Mutex
	rules       map[string]Fault // Filename pattern -> Fault
	defaultErr  error            // Default error for injected faults
	written     int64            // Total bytes written (global)
	globalLimit int64            // Global write limit (-1 = no limit)
}

// NewFaultyFS creates a new FaultyFS wrapping the provided FS.
// If fs is nil, Default (LocalFS) is used.
func NewFaultyFS(fs FileSystem) *FaultyFS {
	if fs == nil {
		fs = Default
	}
	return &FaultyFS{
		FS:          fs,
		rules:       make(map[string]Fault),
		defaultErr:  fmt.Errorf("injected fault error"),
		globalLimit: -1, // No limit by default
	}
}

// SetLimit sets a global byte limit. Writes exceeding this limit will fail.
// Set to -1 to disable the global limit.
func (f *FaultyFS) SetLimit(limit int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.globalLimit = limit
}

// SetError sets the default error returned for injected faults.
func (f *FaultyFS) SetError(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.defaultErr = err
}

// GetWritten returns the total bytes written across all files.
func (f *FaultyFS) GetWritten() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.written
}

// Reset clears the written counter and all rules.
func (f *FaultyFS) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.written = 0
	f.rules = make(map[string]Fault)
}

// AddRule adds a fault injection rule for files matching the pattern.
// Pattern matching uses strings.Contains (substring match).
// Later rules for the same pattern override earlier ones.
func (f *FaultyFS) AddRule(pattern string, fault Fault) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.rules[pattern] = fault
}

func (f *FaultyFS) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	file, err := f.FS.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}

	f.mu.Lock()
	// Find matching rule (last match wins)
	var fault *Fault
	for pattern, rule := range f.rules {
		if strings.Contains(name, pattern) {
			r := rule // Copy to avoid aliasing
			fault = &r
		}
	}
	f.mu.Unlock()

	return &faultyFile{File: file, fs: f, fault: fault}, nil
}

func (f *FaultyFS) Remove(name string) error              { return f.FS.Remove(name) }
func (f *FaultyFS) Rename(oldpath, newpath string) error  { return f.FS.Rename(oldpath, newpath) }
func (f *FaultyFS) Stat(name string) (os.FileInfo, error) { return f.FS.Stat(name) }
func (f *FaultyFS) MkdirAll(path string, perm os.FileMode) error {
	return f.FS.MkdirAll(path, perm)
}
func (f *FaultyFS) ReadDir(name string) ([]os.DirEntry, error) { return f.FS.ReadDir(name) }
func (f *FaultyFS) Truncate(name string, size int64) error     { return f.FS.Truncate(name, size) }

// faultyFile wraps a File to inject faults.
type faultyFile struct {
	File
	fs      *FaultyFS
	fault   *Fault // nil = no per-file fault
	written int64  // bytes written to this file
}

func (ff *faultyFile) Write(p []byte) (n int, err error) {
	writeLen := int64(len(p))

	// Check per-file limit first
	if ff.fault != nil && ff.fault.FailAfterBytes >= 0 {
		if ff.written+writeLen > ff.fault.FailAfterBytes {
			return 0, ff.getError(ff.fault.Err)
		}
	}

	// Check global limit
	ff.fs.mu.Lock()
	globalExceeded := ff.fs.globalLimit >= 0 && ff.fs.written+writeLen > ff.fs.globalLimit
	if !globalExceeded {
		ff.fs.written += writeLen
	}
	ff.fs.mu.Unlock()

	if globalExceeded {
		return 0, ff.getError(nil)
	}

	// Perform actual write
	n, err = ff.File.Write(p)
	if n > 0 {
		ff.written += int64(n)
	}
	return n, err
}

func (ff *faultyFile) WriteAt(p []byte, off int64) (n int, err error) {
	writeLen := int64(len(p))

	// Check per-file limit (based on total written, not position)
	if ff.fault != nil && ff.fault.FailAfterBytes >= 0 {
		if ff.written+writeLen > ff.fault.FailAfterBytes {
			return 0, ff.getError(ff.fault.Err)
		}
	}

	// Check global limit
	ff.fs.mu.Lock()
	globalExceeded := ff.fs.globalLimit >= 0 && ff.fs.written+writeLen > ff.fs.globalLimit
	if !globalExceeded {
		ff.fs.written += writeLen
	}
	ff.fs.mu.Unlock()

	if globalExceeded {
		return 0, ff.getError(nil)
	}

	// Perform actual write
	n, err = ff.File.WriteAt(p, off)
	if n > 0 {
		ff.written += int64(n)
	}
	return n, err
}

func (ff *faultyFile) Sync() error {
	if ff.fault != nil && ff.fault.FailOnSync {
		return ff.getError(ff.fault.Err)
	}
	return ff.File.Sync()
}

func (ff *faultyFile) Close() error {
	if ff.fault != nil && ff.fault.FailOnClose {
		_ = ff.File.Close() // Still close the underlying file
		return ff.getError(ff.fault.Err)
	}
	return ff.File.Close()
}

// getError returns the provided error or the default error.
func (ff *faultyFile) getError(err error) error {
	if err != nil {
		return err
	}
	return ff.fs.defaultErr
}
