package engine

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/internal/conv"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/persistence"
	"github.com/hupe1980/vecgo/wal"
)

var (
	snapshotMagic         = [4]byte{'V', 'G', 'S', '1'}
	snapshotDirMagic      = [4]byte{'V', 'G', 'D', '1'}
	snapshotFooterMagic   = [4]byte{'V', 'G', 'F', '1'}
	snapshotFormatVersion = uint16(2)
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

	headerLen, err := writeHeader(w, codecName)
	if err != nil {
		return err
	}

	cw := &countingWriter{w: w}
	cw.n = headerLen

	idxOff, idxLen, idxChecksum, err := writeIndexSection(cw, idx)
	if err != nil {
		return err
	}

	storeOff, storeLen, storeChecksum, err := writeStoreSection[T](cw, dataStore, func(data T) ([]byte, error) {
		return c.Marshal(data)
	})
	if err != nil {
		return err
	}

	metaOff, metaLen, metaChecksum, err := writeStoreSection[metadata.Metadata](cw, metadataStore, func(meta metadata.Metadata) ([]byte, error) {
		return meta.MarshalBinary()
	})
	if err != nil {
		return err
	}

	// Directory
	dirOff, err := conv.Int64ToUint64(cw.n)
	if err != nil {
		return err
	}
	if err := writeSnapshotDirectory(cw, []snapshotSectionEntry{
		{Type: snapshotSectionIndex, Offset: idxOff, Len: idxLen, Checksum: idxChecksum},
		{Type: snapshotSectionDataStore, Offset: storeOff, Len: storeLen, Checksum: storeChecksum},
		{Type: snapshotSectionMetadataStore, Offset: metaOff, Len: metaLen, Checksum: metaChecksum},
	}); err != nil {
		return err
	}
	cwN, err := conv.Int64ToUint64(cw.n)
	if err != nil {
		return err
	}
	dirLen := cwN - dirOff

	// Footer
	return writeSnapshotFooter(cw, dirOff, dirLen)
}

func writeHeader(w io.Writer, codecName string) (int64, error) {
	var hdr [16]byte
	copy(hdr[0:4], snapshotMagic[:])
	binary.LittleEndian.PutUint16(hdr[4:6], snapshotFormatVersion)
	nameLenU16, err := conv.IntToUint16(len(codecName))
	if err != nil {
		return 0, err
	}
	binary.LittleEndian.PutUint16(hdr[8:10], nameLenU16)
	binary.LittleEndian.PutUint16(hdr[10:12], 3)
	if _, err := w.Write(hdr[:]); err != nil {
		return 0, err
	}
	if codecName != "" {
		if _, err := w.Write([]byte(codecName)); err != nil {
			return 0, err
		}
	}

	currentLen := int64(len(hdr)) + int64(len(codecName))
	padding := (8 - (currentLen % 8)) % 8
	if padding > 0 {
		pad := make([]byte, padding)
		if _, err := w.Write(pad); err != nil {
			return 0, err
		}
	}
	return currentLen + padding, nil
}

func writeIndexSection(cw *countingWriter, idx index.Index) (uint64, uint64, uint32, error) {
	bw, ok := idx.(io.WriterTo)
	if !ok {
		return 0, 0, 0, fmt.Errorf("snapshot: index type %T does not support binary persistence", idx)
	}
	startOff, err := conv.Int64ToUint64(cw.n)
	if err != nil {
		return 0, 0, 0, err
	}
	checksumWriter := persistence.NewChecksumWriter(cw)
	if _, err := bw.WriteTo(checksumWriter); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to write index: %w", err)
	}
	endOff, err := conv.Int64ToUint64(cw.n)
	if err != nil {
		return 0, 0, 0, err
	}
	return startOff, endOff - startOff, checksumWriter.Sum(), nil
}

func writeStoreSection[T any](cw *countingWriter, store Store[T], encode func(T) ([]byte, error)) (uint64, uint64, uint32, error) {
	startOff, err := conv.Int64ToUint64(cw.n)
	if err != nil {
		return 0, 0, 0, err
	}
	checksumWriter := persistence.NewChecksumWriter(cw)

	count, err := conv.IntToUint64(store.Len())
	if err != nil {
		return 0, 0, 0, err
	}
	if err := binary.Write(checksumWriter, binary.LittleEndian, count); err != nil {
		return 0, 0, 0, err
	}

	for id, data := range store.All() {
		dataBytes, err := encode(data)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("failed to encode data for id %d: %w", id, err)
		}

		if err := binary.Write(checksumWriter, binary.LittleEndian, id); err != nil {
			return 0, 0, 0, err
		}
		dbLen, err := conv.IntToUint32(len(dataBytes))
		if err != nil {
			return 0, 0, 0, err
		}
		if err := binary.Write(checksumWriter, binary.LittleEndian, dbLen); err != nil {
			return 0, 0, 0, err
		}
		if _, err := checksumWriter.Write(dataBytes); err != nil {
			return 0, 0, 0, err
		}
	}

	endOff, err := conv.Int64ToUint64(cw.n)
	if err != nil {
		return 0, 0, 0, err
	}
	return startOff, endOff - startOff, checksumWriter.Sum(), nil
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

	c, err = resolveCodec(c, codecName)
	if err != nil {
		return nil, err
	}

	if codecName != "" && c.Name() != codecName {
		return nil, fmt.Errorf("snapshot codec %q does not match provided codec %q", codecName, c.Name())
	}

	idxEntry, ok := sections[snapshotSectionIndex]
	if !ok {
		return nil, fmt.Errorf("snapshot missing index section")
	}
	idx, err := loadIndexSection(r, idxEntry)
	if err != nil {
		return nil, err
	}

	storeEntry, ok := sections[snapshotSectionDataStore]
	if !ok {
		return nil, fmt.Errorf("snapshot missing store section")
	}
	dataStore, err := loadStoreSection[T](r, storeEntry, func(b []byte) (T, error) {
		var data T
		err := c.Unmarshal(b, &data)
		return data, err
	})
	if err != nil {
		return nil, err
	}

	metaEntry, ok := sections[snapshotSectionMetadataStore]
	if !ok {
		return nil, fmt.Errorf("snapshot missing metadata section")
	}
	metaStore, err := loadStoreSection[metadata.Metadata](r, metaEntry, func(b []byte) (metadata.Metadata, error) {
		var meta metadata.Metadata
		err := meta.UnmarshalBinary(b)
		return meta, err
	})
	if err != nil {
		return nil, err
	}

	return &Snapshot[T]{
		Index:         idx,
		DataStore:     dataStore,
		MetadataStore: metaStore,
	}, nil
}

func loadIndexSection(r io.ReadSeeker, entry snapshotSectionEntry) (index.Index, error) {
	off, err := conv.Uint64ToInt64(entry.Offset)
	if err != nil {
		return nil, err
	}
	if _, err := r.Seek(off, io.SeekStart); err != nil {
		return nil, err
	}

	length, err := conv.Uint64ToInt64(entry.Len)
	if err != nil {
		return nil, err
	}
	reader := io.LimitReader(r, length)
	checksumReader := persistence.NewChecksumReader(reader)

	idx, err := index.LoadBinaryIndex(checksumReader)
	if err != nil {
		return nil, err
	}

	if _, err := io.Copy(io.Discard, checksumReader); err != nil {
		return nil, fmt.Errorf("failed to drain index section: %w", err)
	}

	if actualChecksum := checksumReader.Sum(); actualChecksum != entry.Checksum {
		return nil, &persistence.ChecksumMismatchError{
			Expected: entry.Checksum,
			Actual:   actualChecksum,
		}
	}

	return idx, nil
}

func loadStoreSection[T any](r io.ReadSeeker, entry snapshotSectionEntry, decode func([]byte) (T, error)) (Store[T], error) {
	off, err := conv.Uint64ToInt64(entry.Offset)
	if err != nil {
		return nil, err
	}
	if _, err := r.Seek(off, io.SeekStart); err != nil {
		return nil, err
	}

	length, err := conv.Uint64ToInt64(entry.Len)
	if err != nil {
		return nil, err
	}
	reader := io.LimitReader(r, length)
	checksumReader := persistence.NewChecksumReader(reader)

	var count uint64
	if err := binary.Read(checksumReader, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read count: %w", err)
	}
	if count > (entry.Len-8)/12 {
		return nil, fmt.Errorf("count %d exceeds max possible for section size %d", count, entry.Len)
	}

	store := NewMapStore[T]()
	for i := uint64(0); i < count; i++ {
		var id uint32
		if err := binary.Read(checksumReader, binary.LittleEndian, &id); err != nil {
			return nil, fmt.Errorf("failed to read id: %w", err)
		}
		var dataLen uint32
		if err := binary.Read(checksumReader, binary.LittleEndian, &dataLen); err != nil {
			return nil, fmt.Errorf("failed to read data length: %w", err)
		}

		if uint64(dataLen) > entry.Len {
			return nil, fmt.Errorf("data length %d exceeds section length %d", dataLen, entry.Len)
		}

		dataBytes := make([]byte, dataLen)
		if _, err := io.ReadFull(checksumReader, dataBytes); err != nil {
			return nil, fmt.Errorf("failed to read data: %w", err)
		}

		data, err := decode(dataBytes)
		if err != nil {
			return nil, err
		}
		_ = store.Set(core.LocalID(id), data)
	}

	if _, err := io.Copy(io.Discard, checksumReader); err != nil {
		return nil, fmt.Errorf("failed to drain section: %w", err)
	}
	if actualChecksum := checksumReader.Sum(); actualChecksum != entry.Checksum {
		return nil, &persistence.ChecksumMismatchError{
			Expected: entry.Checksum,
			Actual:   actualChecksum,
		}
	}

	return store, nil
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
	entriesLen, err := conv.IntToUint32(len(entries))
	if err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(hdr[8:12], entriesLen)
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
	codecName, sectionCount, err := readSnapshotHeader(r)
	if err != nil {
		return "", nil, err
	}

	dirOffset, err := readSnapshotFooter(r)
	if err != nil {
		return "", nil, err
	}

	sections, err = readSnapshotDirectory(r, dirOffset, sectionCount, len(codecName))
	if err != nil {
		return "", nil, err
	}

	return codecName, sections, nil
}

func readSnapshotHeader(r io.ReadSeeker) (string, int, error) {
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return "", 0, err
	}
	var hdr [16]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return "", 0, err
	}
	if [4]byte(hdr[0:4]) != snapshotMagic {
		return "", 0, fmt.Errorf("unsupported snapshot format: bad magic")
	}
	ver := binary.LittleEndian.Uint16(hdr[4:6])
	if ver != snapshotFormatVersion {
		return "", 0, fmt.Errorf("unsupported snapshot format version: %d", ver)
	}
	nameLen := int(binary.LittleEndian.Uint16(hdr[8:10]))
	if nameLen < 0 || nameLen > 0xFFFF {
		return "", 0, fmt.Errorf("invalid codec name length: %d", nameLen)
	}
	sectionCount := int(binary.LittleEndian.Uint16(hdr[10:12]))
	if sectionCount <= 0 {
		return "", 0, fmt.Errorf("invalid section count: %d", sectionCount)
	}

	nameBytes := make([]byte, nameLen)
	if nameLen > 0 {
		if _, err := io.ReadFull(r, nameBytes); err != nil {
			return "", 0, err
		}
	}
	return string(nameBytes), sectionCount, nil
}

func readSnapshotFooter(r io.ReadSeeker) (int64, error) {
	end, err := r.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}
	if end < 24 {
		return 0, fmt.Errorf("truncated snapshot")
	}
	if _, err := r.Seek(end-24, io.SeekStart); err != nil {
		return 0, err
	}
	var foot [24]byte
	if _, err := io.ReadFull(r, foot[:]); err != nil {
		return 0, err
	}
	if [4]byte(foot[0:4]) != snapshotFooterMagic {
		return 0, fmt.Errorf("unsupported snapshot format: missing footer")
	}
	fver := binary.LittleEndian.Uint16(foot[4:6])
	if fver != snapshotFormatVersion {
		return 0, fmt.Errorf("unsupported snapshot footer version: %d", fver)
	}

	const maxInt64u = ^uint64(0) >> 1
	dirOffsetU := binary.LittleEndian.Uint64(foot[8:16])
	dirLenU := binary.LittleEndian.Uint64(foot[16:24])
	if dirOffsetU > maxInt64u || dirLenU > maxInt64u {
		return 0, fmt.Errorf("invalid directory offsets")
	}
	dataEndU, err := conv.Int64ToUint64(end - 24)
	if err != nil {
		return 0, err
	}
	if dirLenU < 12 || dirOffsetU > dataEndU || dirLenU > dataEndU-dirOffsetU {
		return 0, fmt.Errorf("invalid directory range")
	}

	return conv.Uint64ToInt64(dirOffsetU)
}

func readSnapshotDirectory(r io.ReadSeeker, dirOffset int64, sectionCount int, nameLen int) (map[uint16]snapshotSectionEntry, error) {
	if _, err := r.Seek(dirOffset, io.SeekStart); err != nil {
		return nil, err
	}
	var dh [12]byte
	if _, err := io.ReadFull(r, dh[:]); err != nil {
		return nil, err
	}
	if [4]byte(dh[0:4]) != snapshotDirMagic {
		return nil, fmt.Errorf("invalid snapshot directory magic")
	}
	dver := binary.LittleEndian.Uint16(dh[4:6])
	if dver != snapshotFormatVersion {
		return nil, fmt.Errorf("unsupported snapshot directory version: %d", dver)
	}
	entryCount := int(binary.LittleEndian.Uint32(dh[8:12]))
	if entryCount != sectionCount {
		return nil, fmt.Errorf("directory entry count %d does not match header section count %d", entryCount, sectionCount)
	}

	dirOffsetU, err := conv.Int64ToUint64(dirOffset)
	if err != nil {
		return nil, err
	}

	sections := make(map[uint16]snapshotSectionEntry, entryCount)
	for i := 0; i < entryCount; i++ {
		// v1 format: 32 bytes per entry with checksums
		var eb [32]byte
		if _, err := io.ReadFull(r, eb[:]); err != nil {
			return nil, err
		}
		typ := binary.LittleEndian.Uint16(eb[0:2])
		checksum := binary.LittleEndian.Uint32(eb[4:8])
		off := binary.LittleEndian.Uint64(eb[8:16])
		ln := binary.LittleEndian.Uint64(eb[16:24])
		if _, exists := sections[typ]; exists {
			return nil, fmt.Errorf("duplicate snapshot section type %d", typ)
		}

		// Sections must not point into the header (including codec name).
		headerEndU, err := conv.IntToUint64(16 + nameLen)
		if err != nil {
			return nil, err
		}
		if off < headerEndU {
			return nil, fmt.Errorf("invalid snapshot section offset")
		}
		// Sections must be before the directory.
		if off > dirOffsetU || ln > dirOffsetU-off {
			return nil, fmt.Errorf("invalid snapshot section range")
		}
		sections[typ] = snapshotSectionEntry{Type: typ, Offset: off, Len: ln, Checksum: checksum}
	}
	return sections, nil
}

func resolveCodec(c codec.Codec, codecName string) (codec.Codec, error) {
	if c != nil {
		if codecName != "" && c.Name() != codecName {
			return nil, fmt.Errorf("snapshot codec %q does not match provided codec %q", codecName, c.Name())
		}
		return c, nil
	}

	if codecName == "" {
		return codec.Default, nil
	}

	cc, ok := codec.ByName(codecName)
	if !ok {
		return nil, fmt.Errorf("unsupported snapshot codec %q", codecName)
	}
	return cc, nil
}
