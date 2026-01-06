package engine

import (
	"os"

	"github.com/hupe1980/vecgo/internal/fs"
)

func syncDir(fsys fs.FileSystem, dir string) error {
	f, err := fsys.OpenFile(dir, os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
