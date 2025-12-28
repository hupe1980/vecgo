package engine

import "github.com/hupe1980/vecgo/metadata"

// Durability abstracts the prepare/commit logging used by the mutation coordinator.
//
// When backed by a WAL, these methods persist intent (prepare) and durability
// boundaries (commit). When durability is disabled, a NoopDurability can be used
// to keep the exact same atomic mutation semantics without persistence.
//
// This interface intentionally matches the WAL method surface used by Tx.
// A *wal.WAL satisfies this interface.
type Durability interface {
	LogPrepareInsert(id uint32, vector []float32, payload []byte, meta metadata.Metadata) error
	LogCommitInsert(id uint32) error

	LogPrepareBatchInsert(ids []uint32, vectors [][]float32, payloads [][]byte, metadataSlice []metadata.Metadata) error
	LogCommitBatchInsert(ids []uint32) error

	LogPrepareUpdate(id uint32, vector []float32, payload []byte, meta metadata.Metadata) error
	LogCommitUpdate(id uint32) error

	LogPrepareDelete(id uint32) error
	LogCommitDelete(id uint32) error
}

// NoopDurability implements Durability with no persistence.
//
// Using this keeps the exact same mutation pipeline and rollback behavior
// as WAL-backed mode, but without any disk IO.
//
// This is a breaking-change-friendly way to eliminate the WAL/no-WAL
// correctness bifurcation.
type NoopDurability struct{}

func (NoopDurability) LogPrepareInsert(uint32, []float32, []byte, metadata.Metadata) error {
	return nil
}
func (NoopDurability) LogCommitInsert(uint32) error { return nil }
func (NoopDurability) LogPrepareBatchInsert([]uint32, [][]float32, [][]byte, []metadata.Metadata) error {
	return nil
}
func (NoopDurability) LogCommitBatchInsert([]uint32) error { return nil }
func (NoopDurability) LogPrepareUpdate(uint32, []float32, []byte, metadata.Metadata) error {
	return nil
}
func (NoopDurability) LogCommitUpdate(uint32) error  { return nil }
func (NoopDurability) LogPrepareDelete(uint32) error { return nil }
func (NoopDurability) LogCommitDelete(uint32) error  { return nil }
