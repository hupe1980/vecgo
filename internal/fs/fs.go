package fs

import (
	"io"
	"os"
)

// File represents an open file.
type File interface {
	io.ReadWriteCloser
	io.ReaderAt
	io.Seeker
	Sync() error
	Stat() (os.FileInfo, error)
}

// FileSystem abstracts file system operations for testability.
type FileSystem interface {
	OpenFile(name string, flag int, perm os.FileMode) (File, error)
	Remove(name string) error
	Rename(oldpath, newpath string) error
	Stat(name string) (os.FileInfo, error)
	MkdirAll(path string, perm os.FileMode) error
	ReadDir(name string) ([]os.DirEntry, error)
	Truncate(name string, size int64) error
}

// LocalFS implements FileSystem using the local os package.
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

// Default is the default local file system.
var Default FileSystem = LocalFS{}
