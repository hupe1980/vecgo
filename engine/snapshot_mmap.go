package engine

import (
	"encoding/binary"
	"fmt"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/persistence"
)

// LoadFromFileMmapWithCodec loads a snapshot from a file via mmap.
//
// The returned index may hold zero-copy vector slices that alias the mapped file.
// Call SnapshotMmap.MappedFile.Close() when done to unmap/close the file.
func LoadFromFileMmapWithCodec[T any](filename string, c codec.Codec) (*SnapshotMmap[T], error) {
	mf, err := persistence.MmapReadOnly(filename)
	if err != nil {
		return nil, err
	}
	data := mf.Bytes()

	codecName, sections, err := readSnapshotDirectoryFromBytes(data)
	if err != nil {
		_ = mf.Close()
		return nil, err
	}

	// If the caller didn't provide a codec, infer it from the snapshot header.
	if c == nil {
		if codecName != "" {
			cc, ok := codec.ByName(codecName)
			if !ok {
				_ = mf.Close()
				return nil, fmt.Errorf("unsupported snapshot codec %q", codecName)
			}
			c = cc
		} else {
			c = codec.Default
		}
	}
	if codecName != "" && c.Name() != codecName {
		_ = mf.Close()
		return nil, fmt.Errorf("snapshot codec %q does not match provided codec %q", codecName, c.Name())
	}

	idxEntry, ok := sections[snapshotSectionIndex]
	if !ok {
		_ = mf.Close()
		return nil, fmt.Errorf("snapshot missing index section")
	}
	dataLenU := uint64(len(data))
	if idxEntry.Offset > dataLenU || idxEntry.Len > dataLenU-idxEntry.Offset {
		_ = mf.Close()
		return nil, fmt.Errorf("truncated index section")
	}
	idxBytes := data[idxEntry.Offset : idxEntry.Offset+idxEntry.Len]
	// Verify checksum
	if actualChecksum := persistence.ComputeChecksum(idxBytes); actualChecksum != idxEntry.Checksum {
		_ = mf.Close()
		return nil, &persistence.ChecksumMismatchError{
			Expected: idxEntry.Checksum,
			Actual:   actualChecksum,
		}
	}
	idx, _, err := index.LoadBinaryIndexMmap(idxBytes)
	if err != nil {
		_ = mf.Close()
		return nil, err
	}

	storeEntry, ok := sections[snapshotSectionDataStore]
	if !ok {
		_ = mf.Close()
		return nil, fmt.Errorf("snapshot missing store section")
	}
	if storeEntry.Offset > dataLenU || storeEntry.Len > dataLenU-storeEntry.Offset {
		_ = mf.Close()
		return nil, fmt.Errorf("truncated store section")
	}
	storeBytes := data[storeEntry.Offset : storeEntry.Offset+storeEntry.Len]
	// Verify checksum
	if actualChecksum := persistence.ComputeChecksum(storeBytes); actualChecksum != storeEntry.Checksum {
		_ = mf.Close()
		return nil, &persistence.ChecksumMismatchError{
			Expected: storeEntry.Checksum,
			Actual:   actualChecksum,
		}
	}

	if len(storeBytes) < 8 {
		_ = mf.Close()
		return nil, fmt.Errorf("truncated store section: missing count")
	}
	storeCount := binary.LittleEndian.Uint64(storeBytes)
	storeBytes = storeBytes[8:]

	dataStore := NewMapStore[T]()
	for i := uint64(0); i < storeCount; i++ {
		if len(storeBytes) < 8 { // ID (4) + Len (4)
			_ = mf.Close()
			return nil, fmt.Errorf("truncated store entry header at index %d", i)
		}
		id := binary.LittleEndian.Uint32(storeBytes)
		storeBytes = storeBytes[4:]

		l := binary.LittleEndian.Uint32(storeBytes)
		storeBytes = storeBytes[4:]

		if uint64(len(storeBytes)) < uint64(l) {
			_ = mf.Close()
			return nil, fmt.Errorf("truncated store entry data at index %d", i)
		}
		valBytes := storeBytes[:l]
		storeBytes = storeBytes[l:]

		var val T
		if err := c.Unmarshal(valBytes, &val); err != nil {
			_ = mf.Close()
			return nil, fmt.Errorf("failed to decode data for id %d: %w", id, err)
		}
		if err := dataStore.Set(core.LocalID(id), val); err != nil {
			_ = mf.Close()
			return nil, fmt.Errorf("failed to set data for id %d: %w", id, err)
		}
	}

	metaEntry, ok := sections[snapshotSectionMetadataStore]
	if !ok {
		_ = mf.Close()
		return nil, fmt.Errorf("snapshot missing metadata section")
	}
	if metaEntry.Offset > dataLenU || metaEntry.Len > dataLenU-metaEntry.Offset {
		_ = mf.Close()
		return nil, fmt.Errorf("truncated metadata section")
	}
	metadataBytes := data[metaEntry.Offset : metaEntry.Offset+metaEntry.Len]
	// Verify checksum
	if actualChecksum := persistence.ComputeChecksum(metadataBytes); actualChecksum != metaEntry.Checksum {
		_ = mf.Close()
		return nil, &persistence.ChecksumMismatchError{
			Expected: metaEntry.Checksum,
			Actual:   actualChecksum,
		}
	}

	if len(metadataBytes) < 8 {
		_ = mf.Close()
		return nil, fmt.Errorf("truncated metadata section: missing count")
	}
	metaCount := binary.LittleEndian.Uint64(metadataBytes)
	metadataBytes = metadataBytes[8:]

	metadataStore := NewMapStore[metadata.Metadata]()
	for i := uint64(0); i < metaCount; i++ {
		if len(metadataBytes) < 8 { // ID (4) + Len (4)
			_ = mf.Close()
			return nil, fmt.Errorf("truncated metadata entry header at index %d", i)
		}
		id := binary.LittleEndian.Uint32(metadataBytes)
		metadataBytes = metadataBytes[4:]

		l := binary.LittleEndian.Uint32(metadataBytes)
		metadataBytes = metadataBytes[4:]

		if uint64(len(metadataBytes)) < uint64(l) {
			_ = mf.Close()
			return nil, fmt.Errorf("truncated metadata entry data at index %d", i)
		}
		valBytes := metadataBytes[:l]
		metadataBytes = metadataBytes[l:]

		var meta metadata.Metadata
		if err := meta.UnmarshalBinary(valBytes); err != nil {
			_ = mf.Close()
			return nil, fmt.Errorf("failed to decode metadata for id %d: %w", id, err)
		}
		if err := metadataStore.Set(core.LocalID(id), meta); err != nil {
			_ = mf.Close()
			return nil, fmt.Errorf("failed to set metadata for id %d: %w", id, err)
		}
	}

	return &SnapshotMmap[T]{
		Snapshot: Snapshot[T]{
			Index:         idx,
			DataStore:     dataStore,
			MetadataStore: metadataStore,
		},
		MappedFile: mf,
	}, nil
}

func readSnapshotDirectoryFromBytes(data []byte) (codecName string, sections map[uint16]snapshotSectionEntry, err error) {
	if len(data) < 16+24 {
		return "", nil, fmt.Errorf("truncated snapshot")
	}
	if data[0] != snapshotMagic[0] || data[1] != snapshotMagic[1] || data[2] != snapshotMagic[2] || data[3] != snapshotMagic[3] {
		return "", nil, fmt.Errorf("unsupported snapshot format: bad magic")
	}
	ver := binary.LittleEndian.Uint16(data[4:6])
	if ver != snapshotFormatVersion {
		return "", nil, fmt.Errorf("unsupported snapshot format version: %d", ver)
	}
	nameLen := int(binary.LittleEndian.Uint16(data[8:10]))
	sectionCount := int(binary.LittleEndian.Uint16(data[10:12]))
	off := 16
	if nameLen < 0 || off+nameLen > len(data) {
		return "", nil, fmt.Errorf("truncated snapshot codec name")
	}
	codecName = ""
	if nameLen > 0 {
		codecName = string(data[off : off+nameLen])
		off += nameLen
	}

	// Footer at end
	footerOff := len(data) - 24
	if data[footerOff] != snapshotFooterMagic[0] || data[footerOff+1] != snapshotFooterMagic[1] || data[footerOff+2] != snapshotFooterMagic[2] || data[footerOff+3] != snapshotFooterMagic[3] {
		return "", nil, fmt.Errorf("unsupported snapshot format: missing footer")
	}
	fver := binary.LittleEndian.Uint16(data[footerOff+4 : footerOff+6])
	if fver != snapshotFormatVersion {
		return "", nil, fmt.Errorf("unsupported snapshot footer version: %d", fver)
	}
	dirOffsetU := binary.LittleEndian.Uint64(data[footerOff+8 : footerOff+16])
	dirLenU := binary.LittleEndian.Uint64(data[footerOff+16 : footerOff+24])
	footerOffU := uint64(footerOff)
	if dirLenU < 12 || dirOffsetU > footerOffU || dirLenU > footerOffU-dirOffsetU {
		return "", nil, fmt.Errorf("invalid directory range")
	}
	dirOffset := int(dirOffsetU)
	dirLen := int(dirLenU)

	// Directory
	if data[dirOffset] != snapshotDirMagic[0] || data[dirOffset+1] != snapshotDirMagic[1] || data[dirOffset+2] != snapshotDirMagic[2] || data[dirOffset+3] != snapshotDirMagic[3] {
		return "", nil, fmt.Errorf("invalid snapshot directory magic")
	}
	dver := binary.LittleEndian.Uint16(data[dirOffset+4 : dirOffset+6])
	if dver != snapshotFormatVersion {
		return "", nil, fmt.Errorf("unsupported snapshot directory version: %d", dver)
	}
	entryCount := int(binary.LittleEndian.Uint32(data[dirOffset+8 : dirOffset+12]))
	if entryCount != sectionCount {
		return "", nil, fmt.Errorf("directory entry count %d does not match header section count %d", entryCount, sectionCount)
	}
	pos := dirOffset + 12
	sections = make(map[uint16]snapshotSectionEntry, entryCount)
	for i := 0; i < entryCount; i++ {
		if pos+32 > dirOffset+dirLen {
			return "", nil, fmt.Errorf("truncated snapshot directory")
		}
		typ := binary.LittleEndian.Uint16(data[pos : pos+2])
		checksum := binary.LittleEndian.Uint32(data[pos+4 : pos+8])
		o := binary.LittleEndian.Uint64(data[pos+8 : pos+16])
		ln := binary.LittleEndian.Uint64(data[pos+16 : pos+24])
		if _, exists := sections[typ]; exists {
			return "", nil, fmt.Errorf("duplicate snapshot section type %d", typ)
		}
		// Sections must be within the data area (before the directory) and must not overlap the directory/footer.
		if o > dirOffsetU || ln > dirOffsetU-o {
			return "", nil, fmt.Errorf("invalid snapshot section range")
		}
		// Sections must not point into the header (including codec name).
		if o < uint64(off) {
			return "", nil, fmt.Errorf("invalid snapshot section offset")
		}
		sections[typ] = snapshotSectionEntry{Type: typ, Offset: o, Len: ln, Checksum: checksum}
		pos += 32
	}

	return codecName, sections, nil
}
