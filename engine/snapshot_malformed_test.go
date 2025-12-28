package engine

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestReadSnapshotDirectoryFromSeeker_DuplicateSectionType(t *testing.T) {
	// Build a minimal snapshot with a valid header, directory, and footer.
	// The directory contains two entries of the same type, which should be rejected.
	var b bytes.Buffer

	// Header
	var hdr [16]byte
	copy(hdr[0:4], snapshotMagic[:])
	binary.LittleEndian.PutUint16(hdr[4:6], snapshotFormatVersion)
	binary.LittleEndian.PutUint16(hdr[8:10], 0) // codec name len
	binary.LittleEndian.PutUint16(hdr[10:12], 2)
	if _, err := b.Write(hdr[:]); err != nil {
		t.Fatalf("write header: %v", err)
	}

	// Payload area (sections would live here)
	payload := make([]byte, 64)
	if _, err := b.Write(payload); err != nil {
		t.Fatalf("write payload: %v", err)
	}

	dirOffset := uint64(b.Len())

	// Directory header
	var dh [12]byte
	copy(dh[0:4], snapshotDirMagic[:])
	binary.LittleEndian.PutUint16(dh[4:6], snapshotFormatVersion)
	binary.LittleEndian.PutUint32(dh[8:12], 2)
	if _, err := b.Write(dh[:]); err != nil {
		t.Fatalf("write dir header: %v", err)
	}

	// Two entries, same type
	writeEntry := func(typ uint16, off, ln uint64) {
		var eb [24]byte
		binary.LittleEndian.PutUint16(eb[0:2], typ)
		binary.LittleEndian.PutUint64(eb[8:16], off)
		binary.LittleEndian.PutUint64(eb[16:24], ln)
		if _, err := b.Write(eb[:]); err != nil {
			t.Fatalf("write entry: %v", err)
		}
	}

	writeEntry(snapshotSectionIndex, 16, 8)
	writeEntry(snapshotSectionIndex, 32, 8) // duplicate

	dirLen := uint64(b.Len()) - dirOffset

	// Footer
	var foot [24]byte
	copy(foot[0:4], snapshotFooterMagic[:])
	binary.LittleEndian.PutUint16(foot[4:6], snapshotFormatVersion)
	binary.LittleEndian.PutUint64(foot[8:16], dirOffset)
	binary.LittleEndian.PutUint64(foot[16:24], dirLen)
	if _, err := b.Write(foot[:]); err != nil {
		t.Fatalf("write footer: %v", err)
	}

	r := bytes.NewReader(b.Bytes())
	_, _, err := readSnapshotDirectoryFromSeeker(r)
	if err == nil {
		t.Fatalf("expected error")
	}
}
