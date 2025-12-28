package engine

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/wal"
)

var (
	snapshotMagic         = [4]byte{'V', 'G', 'S', '1'}
	snapshotDirMagic      = [4]byte{'V', 'G', 'D', '1'}
	snapshotFooterMagic   = [4]byte{'V', 'G', 'F', '1'}
	snapshotFormatVersion = uint16(1)
)

const (
	snapshotSectionIndex         = uint16(1)
	snapshotSectionDataStore     = uint16(2)
	snapshotSectionMetadataStore = uint16(3)
)

type snapshotSectionEntry struct {
	Type     uint16
	_        uint16 // reserved
	Offset   uint64
	Len      uint64
	Checksum uint32 // CRC32 checksum of section data
	_        uint32 // reserved for alignment
}

// Snapshot represents a loaded database snapshot.
type Snapshot[T any] struct {
	Index         index.Index
	DataStore     Store[T]
	MetadataStore Store[metadata.Metadata]
}

// SnapshotMmap represents a memory-mapped snapshot with cleanup handle.
type SnapshotMmap[T any] struct {
	Snapshot[T]
	MappedFile *persistence.MappedFile
}

// SaveToWriter saves the database snapshot to w.
//
// Format:
//  1. snapshot header (magic/version/codec)
//  2. index bytes (native binary index stream)
//  3. data store bytes (codec marshaled map)
//  4. metadata store bytes (codec marshaled map)
//  5. directory (offset/length for each section)
//  6. footer (directory offset/length)
func SaveToWriter[T any](w io.Writer, idx index.Index, dataStore Store[T], metadataStore Store[metadata.Metadata], c codec.Codec) error {
	if w == nil {
		return fmt.Errorf("snapshot: writer is nil")
	}
	if idx == nil {
		return fmt.Errorf("snapshot: index is nil")
	}
	if dataStore == nil {
		return fmt.Errorf("snapshot: data store is nil")
	}
	if metadataStore == nil {
		return fmt.Errorf("snapshot: metadata store is nil")
	}
	if c == nil {
		c = codec.Default
	}

	codecName := c.Name()
	if len(codecName) > 0xFFFF {
		return fmt.Errorf("snapshot codec name too long: %d", len(codecName))
	}

	// Header (16 bytes + codec name)
	// [0:4]  magic
	// [4:6]  version
	// [6:8]  flags/reserved
	// [8:10] codec name len
	// [10:12] section count
	// [12:16] reserved
	var hdr [16]byte
	copy(hdr[0:4], snapshotMagic[:])
	binary.LittleEndian.PutUint16(hdr[4:6], snapshotFormatVersion)
	binary.LittleEndian.PutUint16(hdr[8:10], uint16(len(codecName)))
	binary.LittleEndian.PutUint16(hdr[10:12], 3)
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(codecName) > 0 {
		if _, err := w.Write([]byte(codecName)); err != nil {
			return err
		}
	}

	cw := &countingWriter{w: w}
	cw.n = int64(len(hdr)) + int64(len(codecName))

	// Index: native binary persistence with checksum.
	bw, ok := idx.(io.WriterTo)
	if !ok {
		return fmt.Errorf("snapshot: index type %T does not support binary persistence", idx)
	}
	idxOff := uint64(cw.n)
	checksumWriter := persistence.NewChecksumWriter(cw)
	if _, err := bw.WriteTo(checksumWriter); err != nil {
		return fmt.Errorf("failed to write index: %w", err)
	}
	idxLen := uint64(cw.n) - idxOff
	idxChecksum := checksumWriter.Sum()

	// Data store: codec-marshaled map with checksum.
	storeOff := uint64(cw.n)
	storeBytes, err := c.Marshal(dataStore.ToMap())
	if err != nil {
		return fmt.Errorf("failed to encode store: %w", err)
	}
	storeChecksum := persistence.ComputeChecksum(storeBytes)
	if _, err := cw.Write(storeBytes); err != nil {
		return err
	}
	storeLen := uint64(len(storeBytes))

	// Metadata store: codec-marshaled map with checksum.
	metaOff := uint64(cw.n)
	metaBytes, err := c.Marshal(metadataStore.ToMap())
	if err != nil {
		return fmt.Errorf("failed to encode metadata: %w", err)
	}
	metaChecksum := persistence.ComputeChecksum(metaBytes)
	if _, err := cw.Write(metaBytes); err != nil {
		return err
	}
	metaLen := uint64(len(metaBytes))

	// Directory
	dirOff := uint64(cw.n)
	if err := writeSnapshotDirectory(cw, []snapshotSectionEntry{
		{Type: snapshotSectionIndex, Offset: idxOff, Len: idxLen, Checksum: idxChecksum},
		{Type: snapshotSectionDataStore, Offset: storeOff, Len: storeLen, Checksum: storeChecksum},
		{Type: snapshotSectionMetadataStore, Offset: metaOff, Len: metaLen, Checksum: metaChecksum},
	}); err != nil {
		return err
	}
	dirLen := uint64(cw.n) - dirOff

	// Footer
	return writeSnapshotFooter(cw, dirOff, dirLen)
}

// SaveToFile saves a snapshot to filename.
// If walLog is non-nil, it also creates a checkpoint after a successful save.
func SaveToFile[T any](filename string, idx index.Index, dataStore Store[T], metadataStore Store[metadata.Metadata], walLog *wal.WAL, c codec.Codec) error {
	if err := persistence.SaveToFile(filename, func(w io.Writer) error {
		return SaveToWriter(w, idx, dataStore, metadataStore, c)
	}); err != nil {
		return err
	}
	if walLog != nil {
		return walLog.Checkpoint()
	}
	return nil
}

// LoadFromReaderWithCodec loads a database snapshot from r.
//
// The snapshot container requires random access (io.ReadSeeker) so it can
// locate the directory/footer and then parse each section by offset/length.
//
// If c is nil, the codec is inferred from the snapshot header (codec name).
func LoadFromReaderWithCodec[T any](r io.ReadSeeker, c codec.Codec) (*Snapshot[T], error) {
	if r == nil {
		return nil, fmt.Errorf("snapshot: reader is nil")
	}

	codecName, sections, err := readSnapshotDirectoryFromSeeker(r)
	if err != nil {
		return nil, err
	}

	if c == nil {
		if codecName != "" {
			cc, ok := codec.ByName(codecName)
			if !ok {
				return nil, fmt.Errorf("unsupported snapshot codec %q", codecName)
			}
			c = cc
		} else {
			c = codec.Default
		}
	}
	if codecName != "" && c.Name() != codecName {
		return nil, fmt.Errorf("snapshot codec %q does not match provided codec %q", codecName, c.Name())
	}

	idxEntry, ok := sections[snapshotSectionIndex]
	if !ok {
		return nil, fmt.Errorf("snapshot missing index section")
	}
	if _, err := r.Seek(int64(idxEntry.Offset), io.SeekStart); err != nil {
		return nil, err
	}

	// Read index data and verify checksum
	idxData := make([]byte, idxEntry.Len)
	if _, err := io.ReadFull(r, idxData); err != nil {
		return nil, fmt.Errorf("failed to read index section: %w", err)
	}
	if actualChecksum := persistence.ComputeChecksum(idxData); actualChecksum != idxEntry.Checksum {
		return nil, &persistence.ChecksumMismatchError{
			Expected: idxEntry.Checksum,
			Actual:   actualChecksum,
		}
	}

	// Load index from verified data
	idx, err := index.LoadBinaryIndex(bytes.NewReader(idxData))
	if err != nil {
		return nil, err
	}

	storeEntry, ok := sections[snapshotSectionDataStore]
	if !ok {
		return nil, fmt.Errorf("snapshot missing store section")
	}
	if _, err := r.Seek(int64(storeEntry.Offset), io.SeekStart); err != nil {
		return nil, err
	}
	storeBytes := make([]byte, storeEntry.Len)
	if _, err := io.ReadFull(r, storeBytes); err != nil {
		return nil, fmt.Errorf("failed to read store section: %w", err)
	}
	// Verify checksum
	if actualChecksum := persistence.ComputeChecksum(storeBytes); actualChecksum != storeEntry.Checksum {
		return nil, &persistence.ChecksumMismatchError{
			Expected: storeEntry.Checksum,
			Actual:   actualChecksum,
		}
	}
	storeMap := make(map[uint32]T)
	if err := c.Unmarshal(storeBytes, &storeMap); err != nil {
		return nil, fmt.Errorf("failed to decode store: %w", err)
	}
	dataStoreOut := NewMapStore[T]()
	if err := dataStoreOut.BatchSet(storeMap); err != nil {
		return nil, fmt.Errorf("failed to populate store: %w", err)
	}

	metaEntry, ok := sections[snapshotSectionMetadataStore]
	if !ok {
		return nil, fmt.Errorf("snapshot missing metadata section")
	}
	if _, err := r.Seek(int64(metaEntry.Offset), io.SeekStart); err != nil {
		return nil, err
	}
	metaBytes := make([]byte, metaEntry.Len)
	if _, err := io.ReadFull(r, metaBytes); err != nil {
		return nil, fmt.Errorf("failed to read metadata section: %w", err)
	}
	// Verify checksum
	if actualChecksum := persistence.ComputeChecksum(metaBytes); actualChecksum != metaEntry.Checksum {
		return nil, &persistence.ChecksumMismatchError{
			Expected: metaEntry.Checksum,
			Actual:   actualChecksum,
		}
	}
	metadataMap := make(map[uint32]metadata.Metadata)
	if err := c.Unmarshal(metaBytes, &metadataMap); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}
	metadataStoreOut := NewMapStore[metadata.Metadata]()
	if err := metadataStoreOut.BatchSet(metadataMap); err != nil {
		return nil, fmt.Errorf("failed to populate metadata: %w", err)
	}

	return &Snapshot[T]{
		Index:         idx,
		DataStore:     dataStoreOut,
		MetadataStore: metadataStoreOut,
	}, nil
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

func writeSnapshotDirectory(w io.Writer, entries []snapshotSectionEntry) error {
	// Directory header (12 bytes)
	// [0:4] magic
	// [4:6] version
	// [6:8] reserved
	// [8:12] entry count
	var hdr [12]byte
	copy(hdr[0:4], snapshotDirMagic[:])
	binary.LittleEndian.PutUint16(hdr[4:6], snapshotFormatVersion)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(len(entries)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}

	// Each entry is 32 bytes (v2 format with checksums)
	// [0:2] type
	// [2:4] reserved
	// [4:8] checksum (CRC32)
	// [8:16] offset
	// [16:24] length
	// [24:32] reserved
	for _, e := range entries {
		var b [32]byte
		binary.LittleEndian.PutUint16(b[0:2], e.Type)
		binary.LittleEndian.PutUint32(b[4:8], e.Checksum)
		binary.LittleEndian.PutUint64(b[8:16], e.Offset)
		binary.LittleEndian.PutUint64(b[16:24], e.Len)
		if _, err := w.Write(b[:]); err != nil {
			return err
		}
	}
	return nil
}

func writeSnapshotFooter(w io.Writer, dirOffset, dirLen uint64) error {
	// Footer is 24 bytes
	// [0:4] magic
	// [4:6] version
	// [6:8] reserved
	// [8:16] directory offset
	// [16:24] directory length
	var b [24]byte
	copy(b[0:4], snapshotFooterMagic[:])
	binary.LittleEndian.PutUint16(b[4:6], snapshotFormatVersion)
	binary.LittleEndian.PutUint64(b[8:16], dirOffset)
	binary.LittleEndian.PutUint64(b[16:24], dirLen)
	_, err := w.Write(b[:])
	return err
}

func readSnapshotDirectoryFromSeeker(r io.ReadSeeker) (codecName string, sections map[uint16]snapshotSectionEntry, err error) {
	// Header
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return "", nil, err
	}
	var hdr [16]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return "", nil, err
	}
	if [4]byte(hdr[0:4]) != snapshotMagic {
		return "", nil, fmt.Errorf("unsupported snapshot format: bad magic")
	}
	ver := binary.LittleEndian.Uint16(hdr[4:6])
	if ver != snapshotFormatVersion {
		return "", nil, fmt.Errorf("unsupported snapshot format version: %d", ver)
	}
	nameLen := int(binary.LittleEndian.Uint16(hdr[8:10]))
	if nameLen < 0 || nameLen > 0xFFFF {
		return "", nil, fmt.Errorf("invalid codec name length: %d", nameLen)
	}
	sectionCount := int(binary.LittleEndian.Uint16(hdr[10:12]))
	if sectionCount <= 0 {
		return "", nil, fmt.Errorf("invalid section count: %d", sectionCount)
	}

	nameBytes := make([]byte, nameLen)
	if nameLen > 0 {
		if _, err := io.ReadFull(r, nameBytes); err != nil {
			return "", nil, err
		}
	}
	codecName = string(nameBytes)

	// Footer (last 24 bytes)
	end, err := r.Seek(0, io.SeekEnd)
	if err != nil {
		return "", nil, err
	}
	if end < 24 {
		return "", nil, fmt.Errorf("truncated snapshot")
	}
	if _, err := r.Seek(end-24, io.SeekStart); err != nil {
		return "", nil, err
	}
	var foot [24]byte
	if _, err := io.ReadFull(r, foot[:]); err != nil {
		return "", nil, err
	}
	if [4]byte(foot[0:4]) != snapshotFooterMagic {
		return "", nil, fmt.Errorf("unsupported snapshot format: missing footer")
	}
	fver := binary.LittleEndian.Uint16(foot[4:6])
	if fver != snapshotFormatVersion {
		return "", nil, fmt.Errorf("unsupported snapshot footer version: %d", fver)
	}

	const maxInt64u = uint64(^uint64(0) >> 1)
	dirOffsetU := binary.LittleEndian.Uint64(foot[8:16])
	dirLenU := binary.LittleEndian.Uint64(foot[16:24])
	if dirOffsetU > maxInt64u || dirLenU > maxInt64u {
		return "", nil, fmt.Errorf("invalid directory offsets")
	}
	dataEndU := uint64(end - 24)
	if dirLenU < 12 || dirOffsetU > dataEndU || dirLenU > dataEndU-dirOffsetU {
		return "", nil, fmt.Errorf("invalid directory range")
	}

	// Directory header
	if _, err := r.Seek(int64(dirOffsetU), io.SeekStart); err != nil {
		return "", nil, err
	}
	var dh [12]byte
	if _, err := io.ReadFull(r, dh[:]); err != nil {
		return "", nil, err
	}
	if [4]byte(dh[0:4]) != snapshotDirMagic {
		return "", nil, fmt.Errorf("invalid snapshot directory magic")
	}
	dver := binary.LittleEndian.Uint16(dh[4:6])
	if dver != snapshotFormatVersion {
		return "", nil, fmt.Errorf("unsupported snapshot directory version: %d", dver)
	}
	entryCount := int(binary.LittleEndian.Uint32(dh[8:12]))
	if entryCount != sectionCount {
		return "", nil, fmt.Errorf("directory entry count %d does not match header section count %d", entryCount, sectionCount)
	}

	sections = make(map[uint16]snapshotSectionEntry, entryCount)
	for i := 0; i < entryCount; i++ {
		// v1 format: 32 bytes per entry with checksums
		var eb [32]byte
		if _, err := io.ReadFull(r, eb[:]); err != nil {
			return "", nil, err
		}
		typ := binary.LittleEndian.Uint16(eb[0:2])
		checksum := binary.LittleEndian.Uint32(eb[4:8])
		off := binary.LittleEndian.Uint64(eb[8:16])
		ln := binary.LittleEndian.Uint64(eb[16:24])
		if _, exists := sections[typ]; exists {
			return "", nil, fmt.Errorf("duplicate snapshot section type %d", typ)
		}

		// Sections must not point into the header (including codec name).
		headerEndU := uint64(16 + nameLen)
		if off < headerEndU {
			return "", nil, fmt.Errorf("invalid snapshot section offset")
		}
		// Sections must be before the directory.
		if off > dirOffsetU || ln > dirOffsetU-off {
			return "", nil, fmt.Errorf("invalid snapshot section range")
		}
		sections[typ] = snapshotSectionEntry{Type: typ, Offset: off, Len: ln, Checksum: checksum}
	}

	return codecName, sections, nil
}
