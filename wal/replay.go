package wal

import (
	"bufio"
	"errors"
	"fmt"
	"io"

	"github.com/hupe1980/vecgo/core"
)

// ReplayCommitted replays only committed operations.
//
// With the prepare/commit protocol (OpPrepare* + OpCommit*), only operations that
// have a matching commit are applied.
func (w *WAL) ReplayCommitted(callback func(entry Entry) error) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Seek to the start of the entry stream
	if _, err := w.file.Seek(w.dataOffset, 0); err != nil {
		return err
	}

	var reader io.Reader
	if w.compressed {
		if err := w.decompressor.Reset(w.file); err != nil {
			return fmt.Errorf("failed to reset decompressor: %w", err)
		}
		reader = w.decompressor
	} else {
		reader = bufio.NewReader(w.file)
	}

	pendingInsert := map[core.LocalID]Entry{}
	pendingUpdate := map[core.LocalID]Entry{}
	pendingDelete := map[core.LocalID]struct{}{}

	for {
		var entry Entry
		if err := w.decodeEntry(reader, &entry); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("WAL corrupted at entry: %w", err)
		}

		if entry.Type == OpCheckpoint {
			break
		}

		if err := w.processReplayEntry(entry, pendingInsert, pendingUpdate, pendingDelete, callback); err != nil {
			return err
		}
	}

	// Seek back to end for appending
	if _, err := w.file.Seek(0, 2); err != nil {
		return err
	}

	return nil
}

func (w *WAL) processReplayEntry(entry Entry, pendingInsert map[core.LocalID]Entry, pendingUpdate map[core.LocalID]Entry, pendingDelete map[core.LocalID]struct{}, callback func(entry Entry) error) error {
	switch entry.Type {
	case OpPrepareInsert, OpPrepareUpdate, OpPrepareDelete:
		w.processPrepare(entry, pendingInsert, pendingUpdate, pendingDelete)
	case OpCommitInsert, OpCommitUpdate, OpCommitDelete:
		return w.processCommit(entry, pendingInsert, pendingUpdate, pendingDelete, callback)
	default:
		// Ignore legacy or unknown types during committed replay
		// Or should we handle OpInsert/OpUpdate/OpDelete if they exist?
		// If we have mixed log (legacy + new), we should handle them.
		if entry.Type == OpInsert || entry.Type == OpUpdate || entry.Type == OpDelete {
			if err := callback(entry); err != nil {
				return fmt.Errorf("failed to replay entry %d: %w", entry.SeqNum, err)
			}
		}
	}
	return nil
}

func (w *WAL) processPrepare(entry Entry, pendingInsert map[core.LocalID]Entry, pendingUpdate map[core.LocalID]Entry, pendingDelete map[core.LocalID]struct{}) {
	switch entry.Type {
	case OpPrepareInsert:
		pendingInsert[entry.ID] = entry
	case OpPrepareUpdate:
		pendingUpdate[entry.ID] = entry
	case OpPrepareDelete:
		pendingDelete[entry.ID] = struct{}{}
	}
}

func (w *WAL) processCommit(entry Entry, pendingInsert map[core.LocalID]Entry, pendingUpdate map[core.LocalID]Entry, pendingDelete map[core.LocalID]struct{}, callback func(entry Entry) error) error {
	switch entry.Type {
	case OpCommitInsert:
		if prepared, ok := pendingInsert[entry.ID]; ok {
			prepared.Type = OpInsert
			prepared.SeqNum = entry.SeqNum
			if err := callback(prepared); err != nil {
				return fmt.Errorf("failed to replay entry %d: %w", entry.SeqNum, err)
			}
			delete(pendingInsert, entry.ID)
		}
	case OpCommitUpdate:
		if prepared, ok := pendingUpdate[entry.ID]; ok {
			prepared.Type = OpUpdate
			prepared.SeqNum = entry.SeqNum
			if err := callback(prepared); err != nil {
				return fmt.Errorf("failed to replay entry %d: %w", entry.SeqNum, err)
			}
			delete(pendingUpdate, entry.ID)
		}
	case OpCommitDelete:
		if _, ok := pendingDelete[entry.ID]; ok {
			applied := Entry{Type: OpDelete, ID: entry.ID, SeqNum: entry.SeqNum}
			if err := callback(applied); err != nil {
				return fmt.Errorf("failed to replay entry %d: %w", entry.SeqNum, err)
			}
			delete(pendingDelete, entry.ID)
		}
	}
	return nil
}

// Replay replays all operations in the WAL by calling the provided callback.
func (w *WAL) Replay(callback func(entry Entry) error) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Seek to the start of the entry stream
	if _, err := w.file.Seek(w.dataOffset, 0); err != nil {
		return err
	}

	var reader io.Reader
	if w.compressed {
		// Reset decompressor for the file
		if err := w.decompressor.Reset(w.file); err != nil {
			return fmt.Errorf("failed to reset decompressor: %w", err)
		}
		reader = w.decompressor
	} else {
		reader = bufio.NewReader(w.file)
	}

	for {
		var entry Entry
		if err := w.decodeEntry(reader, &entry); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			// Corrupted entry - stop replay
			return fmt.Errorf("WAL corrupted at entry: %w", err)
		}

		// Stop at checkpoint
		if entry.Type == OpCheckpoint {
			break
		}

		// Apply operation
		if err := callback(entry); err != nil {
			return fmt.Errorf("failed to replay entry %d: %w", entry.SeqNum, err)
		}
	}

	// Seek back to end for appending
	if _, err := w.file.Seek(0, 2); err != nil {
		return err
	}

	return nil
}
