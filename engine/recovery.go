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
	if err := validateRecoveryParams(idx, dataStore, metaStore, walLog); err != nil {
		return err
	}
	if c == nil {
		c = codec.Default
	}

	txIdx, ok := idx.(index.TransactionalIndex)
	if !ok {
		return fmt.Errorf("recovery: index type %T must implement index.TransactionalIndex", idx)
	}

	return walLog.ReplayCommitted(func(entry wal.Entry) error {
		return processRecoveryEntry(ctx, entry, txIdx, dataStore, metaStore, c)
	})
}

func validateRecoveryParams[T any](idx index.Index, dataStore Store[T], metaStore *metadata.UnifiedIndex, walLog *wal.WAL) error {
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
	return nil
}

func processRecoveryEntry[T any](ctx context.Context, entry wal.Entry, txIdx index.TransactionalIndex, dataStore Store[T], metaStore *metadata.UnifiedIndex, c codec.Codec) error {
	switch entry.Type {
	case wal.OpInsert:
		return processUpsert(ctx, entry, txIdx, dataStore, metaStore, c, false)
	case wal.OpUpdate:
		return processUpsert(ctx, entry, txIdx, dataStore, metaStore, c, true)
	case wal.OpDelete:
		return processDelete(ctx, entry, txIdx, dataStore, metaStore)
	}
	return nil
}

func processUpsert[T any](ctx context.Context, entry wal.Entry, txIdx index.TransactionalIndex, dataStore Store[T], metaStore *metadata.UnifiedIndex, c codec.Codec, isUpdate bool) error {
	var data T
	if err := c.Unmarshal(entry.Data, &data); err != nil {
		return err
	}

	var err error
	if isUpdate {
		err = txIdx.ApplyUpdate(ctx, entry.ID, entry.Vector)
	} else {
		err = txIdx.ApplyInsert(ctx, entry.ID, entry.Vector)
	}
	if err != nil {
		return err
	}

	if err := dataStore.Set(entry.ID, data); err != nil {
		return err
	}
	if entry.Metadata != nil {
		metaStore.Set(entry.ID, entry.Metadata)
	}
	return nil
}

func processDelete[T any](ctx context.Context, entry wal.Entry, txIdx index.TransactionalIndex, dataStore Store[T], metaStore *metadata.UnifiedIndex) error {
	if err := txIdx.ApplyDelete(ctx, entry.ID); err != nil {
		return err
	}
	_ = dataStore.Delete(entry.ID)
	metaStore.Delete(entry.ID)
	return nil
}
