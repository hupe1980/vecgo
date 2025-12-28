package engine

import (
	"context"
	"fmt"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/wal"
)

// RecoverFromWAL replays the WAL (committed operations only) into the provided
// index and stores.
//
// This is the engine-layer recovery wiring extracted from vecgo.
func RecoverFromWAL[T any](ctx context.Context, idx index.Index, dataStore Store[T], metaStore *metadata.UnifiedIndex, walLog *wal.WAL, c codec.Codec) error {
	if walLog == nil {
		return fmt.Errorf("recovery: WAL is nil")
	}
	if idx == nil {
		return fmt.Errorf("recovery: index is nil")
	}
	if dataStore == nil {
		return fmt.Errorf("recovery: data store is nil")
	}
	if metaStore == nil {
		return fmt.Errorf("recovery: metadata store is nil")
	}
	if c == nil {
		c = codec.Default
	}

	txIdx, ok := idx.(index.TransactionalIndex)
	if !ok {
		return fmt.Errorf("recovery: index type %T must implement index.TransactionalIndex", idx)
	}

	return walLog.ReplayCommitted(func(entry wal.Entry) error {
		switch entry.Type {
		case wal.OpInsert:
			var data T
			if err := c.Unmarshal(entry.Data, &data); err != nil {
				return err
			}

			if err := txIdx.ApplyInsert(ctx, entry.ID, entry.Vector); err != nil {
				return err
			}
			if err := dataStore.Set(entry.ID, data); err != nil {
				return err
			}
			if entry.Metadata != nil {
				metaStore.Set(entry.ID, entry.Metadata)
			}

		case wal.OpUpdate:
			var data T
			if err := c.Unmarshal(entry.Data, &data); err != nil {
				return err
			}

			if err := txIdx.ApplyUpdate(ctx, entry.ID, entry.Vector); err != nil {
				return err
			}
			if err := dataStore.Set(entry.ID, data); err != nil {
				return err
			}
			if entry.Metadata != nil {
				metaStore.Set(entry.ID, entry.Metadata)
			}

		case wal.OpDelete:
			if err := txIdx.ApplyDelete(ctx, entry.ID); err != nil {
				return err
			}
			_ = dataStore.Delete(entry.ID)
			metaStore.Delete(entry.ID)
		}
		return nil
	})
}
