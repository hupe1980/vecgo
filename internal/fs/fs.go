package fs

import (
	"io"
	"os"
)

// File represents an open file with random-access read/write capabilities.
// It embeds standard io interfaces for composability.
type File interface {
	io.ReadWriteCloser
	io.ReaderAt
	io.WriterAt
	io.Seeker
	Sync() error
	Stat() (os.FileInfo, error)
}

// FileSystem abstracts filesystem operations for testability.
// All methods mirror the standard os package with identical semantics.
type FileSystem interface {
	// OpenFile opens a file with the specified flags and permissions.
	// Flags are the same as os.OpenFile (O_RDONLY, O_WRONLY, O_RDWR, O_CREATE, etc.).
	OpenFile(name string, flag int, perm os.FileMode) (File, error)

	// Remove deletes the named file or empty directory.
	Remove(name string) error

	// Rename renames (moves) oldpath to newpath.
	// If newpath exists, it is replaced.
	Rename(oldpath, newpath string) error

	// Stat returns file info for the named file.
	Stat(name string) (os.FileInfo, error)

	// MkdirAll creates a directory and all parent directories.
	MkdirAll(path string, perm os.FileMode) error

	// ReadDir reads the named directory and returns directory entries.
	ReadDir(name string) ([]os.DirEntry, error)

	// Truncate changes the size of the named file.
	Truncate(name string, size int64) error
}

// LocalFS implements FileSystem using the local os package.
// This is the production implementation for local disk access.
type LocalFS struct{}

func (LocalFS) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	return os.OpenFile(name, flag, perm)
}

func (LocalFS) Remove(name string) error              { return os.Remove(name) }
func (LocalFS) Rename(oldpath, newpath string) error  { return os.Rename(oldpath, newpath) }
func (LocalFS) Stat(name string) (os.FileInfo, error) { return os.Stat(name) }
func (LocalFS) MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}
func (LocalFS) ReadDir(name string) ([]os.DirEntry, error) { return os.ReadDir(name) }
func (LocalFS) Truncate(name string, size int64) error     { return os.Truncate(name, size) }

// Default is the default local file system implementation.
var Default FileSystem = LocalFS{}
