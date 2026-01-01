package engine

import (
	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/metadata"
)

// Durability abstracts the prepare/commit logging used by the mutation coordinator.
//
// When backed by a WAL, these methods persist intent (prepare) and durability
// boundaries (commit). When durability is disabled, a NoopDurability can be used
// to keep the exact same atomic mutation semantics without persistence.
//
// This interface intentionally matches the WAL method surface used by Tx.
// A *wal.WAL satisfies this interface.
type Durability interface {
	LogPrepareInsert(id core.LocalID, vector []float32, payload []byte, meta metadata.Metadata) error
	LogCommitInsert(id core.LocalID) error

	LogPrepareBatchInsert(ids []core.LocalID, vectors [][]float32, payloads [][]byte, metadataSlice []metadata.Metadata) error
	LogCommitBatchInsert(ids []core.LocalID) error

	LogPrepareUpdate(id core.LocalID, vector []float32, payload []byte, meta metadata.Metadata) error
	LogCommitUpdate(id core.LocalID) error

	LogPrepareDelete(id core.LocalID) error
	LogCommitDelete(id core.LocalID) error

	// Close releases any resources held by the durability layer (e.g. file handles).
	Close() error
}

// NoopDurability implements Durability with no persistence.
//
// Using this keeps the exact same mutation pipeline and rollback behavior
// as WAL-backed mode, but without any disk IO.
//
// This is a breaking-change-friendly way to eliminate the WAL/no-WAL
// correctness bifurcation.
type NoopDurability struct{}

func (NoopDurability) LogPrepareInsert(core.LocalID, []float32, []byte, metadata.Metadata) error {
	return nil
}
func (NoopDurability) LogCommitInsert(core.LocalID) error { return nil }
func (NoopDurability) LogPrepareBatchInsert([]core.LocalID, [][]float32, [][]byte, []metadata.Metadata) error {
	return nil
}
func (NoopDurability) LogCommitBatchInsert([]core.LocalID) error { return nil }
func (NoopDurability) LogPrepareUpdate(core.LocalID, []float32, []byte, metadata.Metadata) error {
	return nil
}
func (NoopDurability) LogCommitUpdate(core.LocalID) error  { return nil }
func (NoopDurability) LogPrepareDelete(core.LocalID) error { return nil }
func (NoopDurability) LogCommitDelete(core.LocalID) error  { return nil }
func (NoopDurability) Close() error                        { return nil }
