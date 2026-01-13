package fs

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

// Fault defines specific failure behavior.
type Fault struct {
	FailAfterBytes int64 // Fail writes after this many bytes written TO THIS FILE. -1 to disable.
	FailOnSync     bool
	FailOnClose    bool
	Err            error
}

// FaultyFS is a FileSystem wrapper that can inject errors.
type FaultyFS struct {
	FS      FileSystem
	mu      sync.Mutex
	rules   map[string]Fault // Filename pattern -> Fault
	Default Fault            // Fallback

	// Compatibility fields (kept for existing tests)
	Err         error
	written     int64
	globalLimit int64
}

// NewFaultyFS creates a new FaultyFS wrapping the provided FS (or Default if nil).
func NewFaultyFS(fs FileSystem) *FaultyFS {
	if fs == nil {
		fs = Default
	}
	return &FaultyFS{
		FS:    fs,
		rules: make(map[string]Fault),
		Default: Fault{
			FailAfterBytes: -1, // No limit
		},
		Err:         fmt.Errorf("injected fault error"),
		globalLimit: -1,
	}
}

// GetWritten returns the total bytes written so far (compatibility).
func (f *FaultyFS) GetWritten() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.written
}

// SetLimit sets the byte limit (compatibility targeting Global limit).
func (f *FaultyFS) SetLimit(limit int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.globalLimit = limit
}

// AddRule adds a fault injection rule for a specific file pattern.
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
	fault := f.Default
	// Match pattern (last winning match)
	for pattern, rule := range f.rules {
		if strings.Contains(name, pattern) {
			fault = rule // Overwrite simple
		}
	}
	// Propagate compatibility Err if not set in rule
	if fault.Err == nil {
		fault.Err = f.Err
	}
	f.mu.Unlock()

	return &faultyFile{File: file, fs: f, fault: fault}, nil
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

type faultyFile struct {
	File
	fs      *FaultyFS
	fault   Fault
	written int64
}

func (ff *faultyFile) Write(p []byte) (n int, err error) {
	// Check per-file limit FIRST before updating global counter
	if ff.fault.FailAfterBytes >= 0 {
		if ff.written+int64(len(p)) > ff.fault.FailAfterBytes {
			err = ff.fault.Err
			if err == nil {
				err = ff.fs.Err
			}
			if err == nil {
				err = fmt.Errorf("injected fault error")
			}
			return 0, err
		}
	}

	ff.fs.mu.Lock()
	globalExceeded := ff.fs.globalLimit >= 0 && ff.fs.written+int64(len(p)) > ff.fs.globalLimit
	if !globalExceeded {
		ff.fs.written += int64(len(p))
	}
	ff.fs.mu.Unlock()

	if globalExceeded {
		err = ff.fs.Err
		if err == nil {
			err = fmt.Errorf("injected fault error")
		}
		return 0, err
	}

	n, err = ff.File.Write(p)
	if n > 0 {
		ff.written += int64(n)
	}
	return n, err
}

func (ff *faultyFile) Sync() error {
	if ff.fault.FailOnSync {
		err := ff.fault.Err
		if err == nil {
			err = fmt.Errorf("injected sync error")
		}
		return err
	}
	return ff.File.Sync()
}

func (ff *faultyFile) Close() error {
	if ff.fault.FailOnClose {
		err := ff.fault.Err
		if err == nil {
			err = fmt.Errorf("injected close error")
		}
		ff.File.Close()
		return err
	}
	return ff.File.Close()
}
