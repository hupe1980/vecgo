package wal

import (
	"encoding/binary"
	"fmt"
	"io"
	"unsafe"

	"github.com/hupe1980/vecgo/metadata"
)

// encodeEntry writes an entry in binary format.
// Format: [Type:1][SeqNum:8][ID:4][VectorLen:4][Vector:N*4][DataLen:4][Data:N][MetadataLen:4][Metadata:N]
func (w *WAL) encodeEntry(entry *Entry) error {
	// OpInsert/OpUpdate/OpDelete are logical operations emitted by ReplayCommitted,
	// not on-disk entry types.
	if entry.Type == OpInsert || entry.Type == OpUpdate || entry.Type == OpDelete {
		return fmt.Errorf("unsupported on-disk WAL entry type: %v", entry.Type)
	}

	// Write operation type (1 byte)
	if err := binary.Write(w.writer, binary.LittleEndian, entry.Type); err != nil {
		return err
	}

	// Write sequence number (8 bytes)
	if err := binary.Write(w.writer, binary.LittleEndian, entry.SeqNum); err != nil {
		return err
	}

	// Write ID (4 bytes)
	if err := binary.Write(w.writer, binary.LittleEndian, entry.ID); err != nil {
		return err
	}

	// Write vector/data/metadata for operations that carry payload.
	if entry.Type == OpPrepareInsert || entry.Type == OpPrepareUpdate {
		// Vector length (4 bytes)
		vectorLen := uint32(len(entry.Vector)) //nolint:gosec
		if err := binary.Write(w.writer, binary.LittleEndian, vectorLen); err != nil {
			return err
		}

		// Vector data (N * 4 bytes) - zero-copy write
		if vectorLen > 0 {
			byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&entry.Vector[0])), vectorLen*4) //nolint:gosec // unsafe is required for performance
			if _, err := w.writer.Write(byteSlice); err != nil {
				return err
			}
		}

		// Data length (4 bytes)
		dataLen := uint32(len(entry.Data)) //nolint:gosec
		if err := binary.Write(w.writer, binary.LittleEndian, dataLen); err != nil {
			return err
		}

		// Data bytes
		if dataLen > 0 {
			if _, err := w.writer.Write(entry.Data); err != nil {
				return err
			}
		}

		// Metadata (default codec: VecgoBinary)
		var metadataBytes []byte
		if entry.Metadata != nil {
			b, err := entry.Metadata.MarshalBinary()
			if err != nil {
				return err
			}
			metadataBytes = b
		}
		metadataLen := uint32(len(metadataBytes)) //nolint:gosec
		if err := binary.Write(w.writer, binary.LittleEndian, metadataLen); err != nil {
			return err
		}
		if metadataLen > 0 {
			if _, err := w.writer.Write(metadataBytes); err != nil {
				return err
			}
		}
	}

	return nil
}

// decodeEntry reads an entry in binary format.
func (w *WAL) decodeEntry(reader io.Reader, entry *Entry) error {
	// Read operation type (1 byte)
	if err := binary.Read(reader, binary.LittleEndian, &entry.Type); err != nil {
		return err
	}
	if entry.Type == OpInsert || entry.Type == OpUpdate || entry.Type == OpDelete {
		return fmt.Errorf("unsupported WAL entry type (legacy): %v", entry.Type)
	}

	// Read sequence number (8 bytes)
	if err := binary.Read(reader, binary.LittleEndian, &entry.SeqNum); err != nil {
		return err
	}

	// Read ID (4 bytes)
	tmpID := entry.ID
	if err := binary.Read(reader, binary.LittleEndian, &tmpID); err != nil {
		return err
	}
	entry.ID = tmpID

	// Read vector/data/metadata for operations that carry payload.
	if entry.Type == OpPrepareInsert || entry.Type == OpPrepareUpdate {
		// Vector length
		var vectorLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &vectorLen); err != nil {
			return err
		}

		// Vector data
		if vectorLen > 0 {
			entry.Vector = make([]float32, vectorLen)
			byteSlice := unsafe.Slice((*byte)(unsafe.Pointer(&entry.Vector[0])), vectorLen*4) //nolint:gosec // unsafe is required for performance
			if _, err := io.ReadFull(reader, byteSlice); err != nil {
				return err
			}
		}

		// Data length
		var dataLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &dataLen); err != nil {
			return err
		}

		// Data bytes
		if dataLen > 0 {
			entry.Data = make([]byte, dataLen)
			if _, err := io.ReadFull(reader, entry.Data); err != nil {
				return err
			}
		}

		// Metadata length
		var metadataLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &metadataLen); err != nil {
			return err
		}

		// Metadata bytes
		if metadataLen > 0 {
			metadataBytes := make([]byte, metadataLen)
			if _, err := io.ReadFull(reader, metadataBytes); err != nil {
				return err
			}
			var meta metadata.Metadata
			if err := meta.UnmarshalBinary(metadataBytes); err != nil {
				return err
			}
			entry.Metadata = meta
		}
	}

	return nil
}

func (w *WAL) flushLocked() error {
	if err := w.bufWriter.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %w", err)
	}
	if w.compressed {
		if err := w.compressor.Flush(); err != nil {
			return fmt.Errorf("failed to flush compressor: %w", err)
		}
	}
	return nil
}

func (w *WAL) syncCommitLocked() error {
	// Commit is an explicit durability boundary; whether it fsyncs depends on Options.Sync.
	return w.syncIfNeeded()
}
