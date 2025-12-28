package wal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/hupe1980/vecgo/codec"
)

var (
	walMagic          = [4]byte{'V', 'G', 'W', '0'}
	walHeaderVersion  = uint16(2)
	walHeaderFixedLen = 16 // excludes variable codec name bytes
)

type walHeaderInfo struct {
	Compressed        bool
	CompressionLevel  int
	MetadataCodecName string
	HeaderLen         int64
}

func writeWALHeader(w io.Writer, info walHeaderInfo) (int64, error) {
	codecName := info.MetadataCodecName
	if codecName == "" {
		codecName = codec.Default.Name()
	}
	nameBytes := []byte(codecName)
	if len(nameBytes) > 0xFFFF {
		return 0, fmt.Errorf("WAL codec name too long: %d", len(nameBytes))
	}

	var flags uint16
	if info.Compressed {
		flags |= 1
	}
	level := uint8(0)
	if info.Compressed {
		level = uint8(info.CompressionLevel)
	}

	buf := make([]byte, 0, walHeaderFixedLen+len(nameBytes))
	buf = append(buf, walMagic[:]...)
	var fixed [12]byte
	binary.LittleEndian.PutUint16(fixed[0:2], walHeaderVersion)
	binary.LittleEndian.PutUint16(fixed[2:4], flags)
	fixed[4] = level
	// fixed[5:8] reserved
	binary.LittleEndian.PutUint16(fixed[8:10], uint16(len(nameBytes)))
	// fixed[10:12] reserved
	buf = append(buf, fixed[:]...)
	buf = append(buf, nameBytes...)

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
	nameLen := int(binary.LittleEndian.Uint16(fixed[8:10]))

	nameBytes := make([]byte, nameLen)
	if nameLen > 0 {
		if _, err := io.ReadFull(f, nameBytes); err != nil {
			return walHeaderInfo{}, true, fmt.Errorf("failed to read WAL codec name: %w", err)
		}
	}

	headerLen := int64(4) + int64(len(fixed)) + int64(nameLen)
	if string(bytes.TrimSpace(nameBytes)) == "" {
		return walHeaderInfo{}, true, fmt.Errorf("WAL header codec name is empty")
	}
	return walHeaderInfo{
		Compressed:        compressed,
		CompressionLevel:  level,
		MetadataCodecName: string(bytes.TrimSpace(nameBytes)),
		HeaderLen:         headerLen,
	}, true, nil
}
