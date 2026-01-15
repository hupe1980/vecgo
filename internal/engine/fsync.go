package engine

import (
	"os"

	"github.com/hupe1980/vecgo/internal/fs"
)

// syncDir syncs a directory to ensure metadata changes are persisted.
// This is important for durability after creating/renaming files.
func syncDir(fsys fs.FileSystem, dir string) error {
	f, err := fsys.OpenFile(dir, os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }() // Intentionally ignore: Sync is the important operation
	return f.Sync()
}

// Ensure syncDir is used (prevents unused lint error).
// This function is kept for durability in local mode but currently
// the cloud-native architecture uses BlobStore.Put which handles atomicity.
var _ = syncDir
