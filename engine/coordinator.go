// Package engine provides the coordination layer for vecgo.
//
// Vecgo routes all mutations through a Coordinator/Tx to provide atomic
// multi-subsystem semantics:
//   - (optional) durability prepare entry
//   - apply index mutation by explicit ID (deterministic)
//   - write payload + metadata stores + update meta index
//   - (optional) durability commit entry (durability boundary)
//
// Recovery ignores prepares without commits.
package engine

import (
	"fmt"

	"github.com/hupe1980/vecgo/codec"
	"github.com/hupe1980/vecgo/index"
	"github.com/hupe1980/vecgo/metadata"
)

// Coordinator is the compatibility surface used by vecgo today.
//
// The implementation lives in Tx and is exposed via a type alias so we can
// incrementally split engine concerns (tx/recovery/snapshot/...) without
// changing vecgo's wiring.
type Coordinator[T any] = Tx[T]

// New constructs a Coordinator.
//
// If d is nil, NoopDurability is used (same atomicity semantics, no persistence).
// The index MUST implement index.TransactionalIndex.
func New[T any](idx index.Index, dataStore Store[T], metaStore *metadata.UnifiedIndex, d Durability, c codec.Codec) (*Coordinator[T], error) {
	if idx == nil {
		return nil, fmt.Errorf("coordinator: index is nil")
	}
	if dataStore == nil {
		return nil, fmt.Errorf("coordinator: data store is nil")
	}
	if metaStore == nil {
		return nil, fmt.Errorf("coordinator: metadata store is nil")
	}
	if c == nil {
		c = codec.Default
	}
	if d == nil {
		d = NoopDurability{}
	}

	// Require TransactionalIndex
	txIdx, ok := idx.(index.TransactionalIndex)
	if !ok {
		return nil, fmt.Errorf("coordinator: index type %T must implement index.TransactionalIndex", idx)
	}

	return &Tx[T]{
		txIndex:    txIdx,
		dataStore:  dataStore,
		metaStore:  metaStore,
		durability: d,
		codec:      c,
	}, nil
}
