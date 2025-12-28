package engine

import (
	"context"
	"fmt"
	"sync"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// Tx is the transaction/coordination unit for WAL-backed mutations.
//
// In the current implementation, Tx provides the concrete behavior while
// Coordinator is a compatibility alias used by vecgo.
type Tx[T any] struct {
	mu sync.Mutex

	// TransactionalIndex provides consolidated ID allocation, apply ops, and vector access
	txIndex index.TransactionalIndex

	dataStore Store[T]
	metaStore *metadata.UnifiedIndex

	durability Durability
	codec      codec.Codec
}

// allocateID allocates a new ID from the index
func (tx *Tx[T]) allocateID() uint32 {
	return tx.txIndex.AllocateID()
}

// releaseID releases an ID back to the index
func (tx *Tx[T]) releaseID(id uint32) {
	tx.txIndex.ReleaseID(id)
}

// applyInsert applies an insert operation to the index
func (tx *Tx[T]) applyInsert(ctx context.Context, id uint32, vector []float32) error {
	return tx.txIndex.ApplyInsert(ctx, id, vector)
}

// applyUpdate applies an update operation to the index
func (tx *Tx[T]) applyUpdate(ctx context.Context, id uint32, vector []float32) error {
	return tx.txIndex.ApplyUpdate(ctx, id, vector)
}

// applyDelete applies a delete operation to the index
func (tx *Tx[T]) applyDelete(ctx context.Context, id uint32) error {
	return tx.txIndex.ApplyDelete(ctx, id)
}

// vectorByID retrieves a vector from the index by ID
func (tx *Tx[T]) vectorByID(ctx context.Context, id uint32) ([]float32, error) {
	return tx.txIndex.VectorByID(ctx, id)
}

func (tx *Tx[T]) encodePayload(data T) ([]byte, error) {
	return tx.codec.Marshal(data)
}

// Insert inserts a new vector+payload+(optional) metadata atomically.
func (tx *Tx[T]) Insert(ctx context.Context, vector []float32, data T, meta metadata.Metadata) (uint32, error) {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return 0, err
	}

	id := tx.allocateID()
	payload, err := tx.encodePayload(data)
	if err != nil {
		tx.releaseID(id)
		return 0, err
	}

	if err := tx.durability.LogPrepareInsert(id, vector, payload, meta); err != nil {
		tx.releaseID(id)
		return 0, err
	}

	if err := tx.applyInsert(ctx, id, vector); err != nil {
		tx.releaseID(id)
		return 0, err
	}

	if err := tx.dataStore.Set(id, data); err != nil {
		_ = tx.applyDelete(ctx, id)
		return 0, err
	}

	if meta != nil {
		tx.metaStore.Set(id, meta)
	}

	if err := tx.durability.LogCommitInsert(id); err != nil {
		if meta != nil {
			tx.metaStore.Delete(id)
		}
		_ = tx.dataStore.Delete(id)
		_ = tx.applyDelete(ctx, id)
		return 0, err
	}

	return id, nil
}

// BatchInsert inserts multiple vectors+payloads+(optional) metadata atomically.
func (tx *Tx[T]) BatchInsert(ctx context.Context, vectors [][]float32, dataSlice []T, metadataSlice []metadata.Metadata) ([]uint32, error) {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(vectors) != len(dataSlice) || len(vectors) != len(metadataSlice) {
		return nil, fmt.Errorf("batch size mismatch: vectors=%d data=%d metadata=%d", len(vectors), len(dataSlice), len(metadataSlice))
	}
	if len(vectors) == 0 {
		return nil, nil
	}

	ids := make([]uint32, len(vectors))
	payloads := make([][]byte, len(vectors))
	for i := range vectors {
		ids[i] = tx.allocateID()
		b, err := tx.encodePayload(dataSlice[i])
		if err != nil {
			for j := 0; j <= i; j++ {
				tx.releaseID(ids[j])
			}
			return nil, err
		}
		payloads[i] = b
	}

	if err := tx.durability.LogPrepareBatchInsert(ids, vectors, payloads, metadataSlice); err != nil {
		for _, id := range ids {
			tx.releaseID(id)
		}
		return nil, err
	}

	for i := range vectors {
		if err := tx.applyInsert(ctx, ids[i], vectors[i]); err != nil {
			for j := 0; j < i; j++ {
				_ = tx.applyDelete(ctx, ids[j])
			}
			for j := i; j < len(ids); j++ {
				tx.releaseID(ids[j])
			}
			return nil, err
		}
	}

	items := make(map[uint32]T, len(ids))
	for i := range ids {
		items[ids[i]] = dataSlice[i]
	}
	if err := tx.dataStore.BatchSet(items); err != nil {
		for _, id := range ids {
			_ = tx.applyDelete(ctx, id)
		}
		return nil, err
	}

	metaItems := make(map[uint32]metadata.Metadata)
	for i := range ids {
		if metadataSlice[i] != nil {
			metaItems[ids[i]] = metadataSlice[i]
		}
	}
	if len(metaItems) > 0 {
		for id, doc := range metaItems {
			tx.metaStore.Set(id, doc)
		}
	}

	if err := tx.durability.LogCommitBatchInsert(ids); err != nil {
		if len(metaItems) > 0 {
			for id := range metaItems {
				tx.metaStore.Delete(id)
			}
		}
		_ = tx.dataStore.BatchDelete(ids)
		for _, id := range ids {
			_ = tx.applyDelete(ctx, id)
		}
		return nil, err
	}

	return ids, nil
}

// Update updates vector+payload and optionally metadata.
// If meta is nil, metadata is left unchanged (matches Vecgo.Update behavior).
func (tx *Tx[T]) Update(ctx context.Context, id uint32, vector []float32, data T, meta metadata.Metadata) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return err
	}

	oldVector, err := tx.vectorByID(ctx, id)
	if err != nil {
		return err
	}
	oldData, ok := tx.dataStore.Get(id)
	if !ok {
		return ErrNotFound
	}

	var oldMeta metadata.Metadata
	oldMetaOK := false
	if meta != nil {
		oldMeta, oldMetaOK = tx.metaStore.Get(id)
	}

	payload, err := tx.encodePayload(data)
	if err != nil {
		return err
	}

	if err := tx.durability.LogPrepareUpdate(id, vector, payload, meta); err != nil {
		return err
	}

	if err := tx.applyUpdate(ctx, id, vector); err != nil {
		return err
	}

	if err := tx.dataStore.Set(id, data); err != nil {
		_ = tx.applyUpdate(ctx, id, oldVector)
		return err
	}

	if meta != nil {
		tx.metaStore.Set(id, meta)
	}

	if err := tx.durability.LogCommitUpdate(id); err != nil {
		if meta != nil {
			if oldMetaOK {
				tx.metaStore.Set(id, oldMeta)
			} else {
				tx.metaStore.Delete(id)
			}
		}
		_ = tx.dataStore.Set(id, oldData)
		_ = tx.applyUpdate(ctx, id, oldVector)
		return err
	}

	return nil
}

// Delete removes a vector and associated data from the database.
func (tx *Tx[T]) Delete(ctx context.Context, id uint32) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return err
	}

	oldVector, err := tx.vectorByID(ctx, id)
	if err != nil {
		return err
	}
	oldData, ok := tx.dataStore.Get(id)
	if !ok {
		return ErrNotFound
	}
	oldMeta, oldMetaOK := tx.metaStore.Get(id)

	if err := tx.durability.LogPrepareDelete(id); err != nil {
		return err
	}

	if err := tx.applyDelete(ctx, id); err != nil {
		return err
	}

	if err := tx.dataStore.Delete(id); err != nil {
		_ = tx.applyInsert(ctx, id, oldVector)
		return err
	}
	if oldMetaOK {
		tx.metaStore.Delete(id)
	}

	if err := tx.durability.LogCommitDelete(id); err != nil {
		_ = tx.applyInsert(ctx, id, oldVector)
		_ = tx.dataStore.Set(id, oldData)
		if oldMetaOK {
			tx.metaStore.Set(id, oldMeta)
		}
		return err
	}

	return nil
}

// KNNSearch performs a K-nearest neighbor search on the underlying index.
// This method is added to satisfy the coordinator[T] interface.
func (tx *Tx[T]) KNNSearch(ctx context.Context, query []float32, k int, opts *index.SearchOptions) ([]index.SearchResult, error) {
	return tx.txIndex.KNNSearch(ctx, query, k, opts)
}

// BruteSearch performs a brute-force search on the underlying index.
// This method is added to satisfy the coordinator[T] interface.
func (tx *Tx[T]) BruteSearch(ctx context.Context, query []float32, k int, filter func(id uint32) bool) ([]index.SearchResult, error) {
	return tx.txIndex.BruteSearch(ctx, query, k, filter)
}
