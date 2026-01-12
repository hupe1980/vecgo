package pk

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/hupe1980/vecgo/model"
)

const (
	magicPKIndex = 0x504B4958 // "PKIX"
	versionPK    = 1
)

// Save persists the index specific to the current state (latest versions).
// It does NOT persist full MVCC history, only the latest version for each key.
// effectively creating a checkpoint at the current moment.
func (idx *Index) Save(w io.Writer) error {
	bw := bufio.NewWriter(w)
	defer bw.Flush()

	// Header
	var buf [16]byte
	binary.LittleEndian.PutUint32(buf[0:4], magicPKIndex)
	binary.LittleEndian.PutUint32(buf[4:8], versionPK)
	binary.LittleEndian.PutUint64(buf[8:16], uint64(idx.count.Load()))
	if _, err := bw.Write(buf[:]); err != nil {
		return err
	}

	// Pages
	pages := *idx.pages.Load()
	// Write number of pages
	if err := binary.Write(bw, binary.LittleEndian, uint32(len(pages))); err != nil {
		return err
	}

	entryBuf := make([]byte, 20) // SegID(8) + RowID(4) + LSN(8)

	// Iterate pages
	for _, p := range pages {
		// Write page existence flag
		exists := p != nil
		if err := binary.Write(bw, binary.LittleEndian, exists); err != nil {
			return err
		}
		if !exists {
			continue
		}

		// Write entries for the page
		// We lock entries individually? No, Save is usually called during checkpoint where we might have a global lock or we race.
		// If we want a consistent snapshot, we should probably rely on the caller to ensure consistency or use MVCC mechanics.
		// For simplicity/speed in this pass: we just take the HEAD.
		// NOTE: In production, we should probably respect a snapshot LSN.
		// But here, let's assume we save the "Current" state.

		for j := 0; j < pageSize; j++ {
			head := p.entries[j].head.Load()
			// Flatten: only keep the head (latest)

			var segID uint64
			var rowID uint32
			var lsn uint64
			var deleted byte

			if head != nil {
				segID = uint64(head.location.SegmentID)
				rowID = uint32(head.location.RowID)
				lsn = head.lsn
				deleted = 2 // Bit 1: Exists
				if head.deleted {
					deleted |= 1 // Bit 0: Deleted
				}
			}

			binary.LittleEndian.PutUint64(entryBuf[0:8], segID)
			binary.LittleEndian.PutUint32(entryBuf[8:12], rowID)
			binary.LittleEndian.PutUint64(entryBuf[12:20], lsn)

			if _, err := bw.Write(entryBuf); err != nil {
				return err
			}
			if err := bw.WriteByte(deleted); err != nil {
				return err
			}
		}
	}

	return nil
}

// Load restores the index from the writer.
// It assumes the index is empty.
func (idx *Index) Load(r io.Reader) error {
	br := bufio.NewReader(r)

	var buf [16]byte
	if _, err := io.ReadFull(br, buf[:]); err != nil {
		return err
	}

	magic := binary.LittleEndian.Uint32(buf[0:4])
	if magic != magicPKIndex {
		return fmt.Errorf("invalid magic: %x", magic)
	}
	ver := binary.LittleEndian.Uint32(buf[4:8])
	if ver != versionPK {
		return fmt.Errorf("unsupported version: %d", ver)
	}
	count := binary.LittleEndian.Uint64(buf[8:16])
	idx.count.Store(int64(count))

	var numPages uint32
	if err := binary.Read(br, binary.LittleEndian, &numPages); err != nil {
		return err
	}

	newPages := make([]*page, numPages)
	entryBuf := make([]byte, 20)

	for i := 0; i < int(numPages); i++ {
		var exists bool
		if err := binary.Read(br, binary.LittleEndian, &exists); err != nil {
			return err
		}
		if !exists {
			continue
		}

		p := &page{}
		for j := 0; j < pageSize; j++ {
			if _, err := io.ReadFull(br, entryBuf); err != nil {
				return err
			}
			deletedByte, err := br.ReadByte()
			if err != nil {
				return err
			}

			segID := binary.LittleEndian.Uint64(entryBuf[0:8])
			rowID := binary.LittleEndian.Uint32(entryBuf[8:12])
			lsn := binary.LittleEndian.Uint64(entryBuf[12:20])

			exists := (deletedByte & 2) != 0
			if exists {
				p.entries[j].head.Store(&version{
					lsn: lsn,
					location: model.Location{
						SegmentID: model.SegmentID(segID),
						RowID:     model.RowID(rowID),
					},
					deleted: (deletedByte & 1) != 0,
				})
			}
		}
		newPages[i] = p
	}

	idx.pages.Store(&newPages)
	return nil
}
