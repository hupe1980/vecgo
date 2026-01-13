package engine

import (
	"context"
	"fmt"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// OpType represents the type of operation in a batch.
type OpType uint8

const (
	OpInsert OpType = iota
	OpDelete
)

// BatchOp represents a single operation in a WriteBatch.
type BatchOp struct {
	Type OpType
	// ID is used for Delete. Ignored for Insert (auto-generated).
	ID       model.ID
	Vector   []float32
	Metadata metadata.Document
	Payload  []byte
}

// WriteBatch accumulates a batch of operations to be executed atomically.
// It is not thread-safe and should be used by a single goroutine.
type WriteBatch struct {
	ops []BatchOp
}

// NewWriteBatch creates a new empty WriteBatch.
func NewWriteBatch() *WriteBatch {
	return &WriteBatch{}
}

// AddInsert adds an insert operation to the batch.
func (b *WriteBatch) AddInsert(vector []float32, md metadata.Document, payload []byte) {
	b.ops = append(b.ops, BatchOp{
		Type:     OpInsert,
		Vector:   vector,
		Metadata: md,
		Payload:  payload,
	})
}

// AddDelete adds a delete operation to the batch.
func (b *WriteBatch) AddDelete(id model.ID) {
	b.ops = append(b.ops, BatchOp{
		Type: OpDelete,
		ID:   id,
	})
}

// Len returns the number of operations in the batch.
func (b *WriteBatch) Len() int {
	return len(b.ops)
}

// Clear resets the batch for reuse.
func (b *WriteBatch) Clear() {
	b.ops = b.ops[:0]
}

// ApplyBatch executes a batch of operations atomically.
// It returns the IDs of inserted records (0 for non-inserts) and any error.
func (e *Engine) ApplyBatch(ctx context.Context, batch *WriteBatch) ([]model.ID, error) {
	n := batch.Len()
	if n == 0 {
		return nil, nil
	}

	// Validate inserts
	for i, op := range batch.ops {
		if op.Type == OpInsert {
			if len(op.Vector) != e.dim {
				return nil, fmt.Errorf("op %d: %w: expected %d, got %d", i, ErrInvalidArgument, e.dim, len(op.Vector))
			}
			if e.schema != nil && op.Metadata != nil {
				if err := e.schema.Validate(op.Metadata); err != nil {
					return nil, fmt.Errorf("op %d: %w: %w", i, ErrInvalidArgument, err)
				}
			}
		}
	}

	e.mu.RLock()
	if e.closed.Load() {
		e.mu.RUnlock()
		return nil, ErrClosed
	}

	snap := e.current.Load()
	resultIDs := make([]model.ID, n)

	for i, op := range batch.ops {
		lsn := e.lsn.Add(1)
		var id model.ID

		if op.Type == OpInsert {
			// Generate ID
			id = model.ID(e.nextID.Add(1))
			resultIDs[i] = id
		} else {
			id = op.ID
			// Deletes don't generate new IDs
		}

		// Apply to Memory
		if op.Type == OpInsert {
			rowID, err := snap.active.InsertWithPayload(id, op.Vector, op.Metadata, op.Payload)
			if err != nil {
				e.mu.RUnlock()
				return resultIDs, err
			}

			oldLoc, exists := e.pkIndex.Upsert(id, model.Location{
				SegmentID: snap.active.ID(),
				RowID:     rowID,
			}, lsn)

			if exists {
				// Mark old location as deleted in tombstones
				// Note: tombstones map is pre-populated for all known segments
				if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
					vt.MarkDeleted(uint32(oldLoc.RowID), lsn)
				}
			}

			// Lexical Update
			if e.lexicalIndex != nil && e.lexicalField != "" && op.Metadata != nil {
				if val, ok := op.Metadata[e.lexicalField]; ok {
					if str := val.StringValue(); str != "" {
						_ = e.lexicalIndex.Add(id, str)
					}
				}
			}
		} else if op.Type == OpDelete {
			// Delete logic
			oldLoc, exists := e.pkIndex.Delete(id, lsn)
			if exists {
				if vt, ok := e.tombstones[oldLoc.SegmentID]; ok {
					vt.MarkDeleted(uint32(oldLoc.RowID), lsn)
				}
				// Also remove from lexical index
				if e.lexicalIndex != nil {
					_ = e.lexicalIndex.Delete(id)
				}
			}
			// ID not found is ok - idempotent delete
		}
	}

	// Check triggers
	memSize := snap.active.Size()
	shouldFlush := memSize > e.flushConfig.MaxMemTableSize

	if e.flushConfig.MaxMemTableSize > 0 {
		e.metrics.OnMemTableStatus(memSize, float64(memSize)/float64(e.flushConfig.MaxMemTableSize))
	}

	e.mu.RUnlock()

	if shouldFlush {
		select {
		case e.flushCh <- struct{}{}:
		default:
		}
	}

	return resultIDs, nil
}
