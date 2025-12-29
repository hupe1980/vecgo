package wal

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

var (
	walMagic          = [4]byte{'V', 'G', 'W', '0'}
	walHeaderVersion  = uint16(1)
	walHeaderFixedLen = 16 // excludes variable codec name bytes
)

type walHeaderInfo struct {
	Compressed       bool
	CompressionLevel int
	HeaderLen        int64
}

func writeWALHeader(w io.Writer, info walHeaderInfo) (int64, error) {
	var flags uint16
	if info.Compressed {
		flags |= 1
	}
	level := uint8(0)
	if info.Compressed {
		level = uint8(info.CompressionLevel)
	}

	buf := make([]byte, 0, walHeaderFixedLen)
	buf = append(buf, walMagic[:]...)
	var fixed [12]byte
	binary.LittleEndian.PutUint16(fixed[0:2], walHeaderVersion)
	binary.LittleEndian.PutUint16(fixed[2:4], flags)
	fixed[4] = level
	// fixed[5:12] reserved
	buf = append(buf, fixed[:]...)

	if _, err := w.Write(buf); err != nil {
		return 0, fmt.Errorf("failed to write WAL header: %w", err)
	}
	return int64(len(buf)), nil
}

func readWALHeader(f *os.File) (walHeaderInfo, bool, error) {
	if _, err := f.Seek(0, 0); err != nil {
		return walHeaderInfo{}, false, fmt.Errorf("failed to seek WAL: %w", err)
	}

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		if err == io.EOF {
			return walHeaderInfo{}, false, nil
		}
		return walHeaderInfo{}, false, fmt.Errorf("failed to read WAL header magic: %w", err)
	}
	if magic != walMagic {
		return walHeaderInfo{}, false, fmt.Errorf("unsupported WAL format: invalid header magic")
	}

	fixed := make([]byte, walHeaderFixedLen-4)
	if _, err := io.ReadFull(f, fixed); err != nil {
		return walHeaderInfo{}, true, fmt.Errorf("failed to read WAL header: %w", err)
	}

	version := binary.LittleEndian.Uint16(fixed[0:2])
	if version != walHeaderVersion {
		return walHeaderInfo{}, true, fmt.Errorf("unsupported WAL header version: %d", version)
	}
	flags := binary.LittleEndian.Uint16(fixed[2:4])
	compressed := (flags & 1) != 0
	level := int(fixed[4])
	// fixed[5:12] reserved

	headerLen := int64(walHeaderFixedLen)
	return walHeaderInfo{
		Compressed:       compressed,
		CompressionLevel: level,
		HeaderLen:        headerLen,
	}, true, nil
}
