package wal

import (
	"time"

	"github.com/hupe1980/vecgo/core"
	"github.com/hupe1980/vecgo/metadata"
)

// DurabilityMode defines the fsync behavior for WAL writes.
type DurabilityMode int

const (
	// DurabilityAsync represents asynchronous durability.
	// No fsync, fastest writes but risk of data loss on crash.
	// Use for non-critical workloads or when external replication provides durability.
	DurabilityAsync DurabilityMode = iota

	// DurabilityGroupCommit represents group commit durability.
	// Batched fsync at regular intervals.
	// Balances throughput and durability by amortizing fsync cost across multiple operations.
	// Recommended for most production workloads.
	DurabilityGroupCommit

	// DurabilitySync represents synchronous durability.
	// fsync after every operation.
	// Slowest but strongest durability guarantee. Use for critical data.
	DurabilitySync
)

// OperationType represents the type of operation in the WAL.
type OperationType uint8

const (
	// OpInsert represents an insert operation.
	OpInsert OperationType = iota
	// OpUpdate represents an update operation.
	OpUpdate
	// OpDelete represents a delete operation.
	OpDelete
	// OpCheckpoint represents a checkpoint marker.
	OpCheckpoint

	// Prepare/Commit protocol (atomic recovery):
	// A Prepare entry records the intended mutation; a Commit entry marks it as durable.
	// Recovery must apply only committed operations.

	// OpPrepareInsert represents a prepare insert operation.
	OpPrepareInsert
	// OpPrepareUpdate represents a prepare update operation.
	OpPrepareUpdate
	// OpPrepareDelete represents a prepare delete operation.
	OpPrepareDelete
	// OpCommitInsert represents a commit insert operation.
	OpCommitInsert
	// OpCommitUpdate represents a commit update operation.
	OpCommitUpdate
	// OpCommitDelete represents a commit delete operation.
	OpCommitDelete
)

// Entry represents a single entry in the WAL.
type Entry struct {
	Type     OperationType
	ID       core.LocalID
	Vector   []float32
	Data     []byte // Serialized user data
	Metadata metadata.Metadata
	SeqNum   uint64 // Sequence number for ordering
}

// Options contains configuration for the WAL.
type Options struct {
	// Path is the directory where WAL files are stored.
	Path string

	// Compress enables zstd compression (2-3x smaller, slightly slower writes).
	// Recommended for production use to reduce disk I/O and storage costs.
	Compress bool

	// CompressionLevel sets the zstd compression level (1-22).
	// Default (3) provides good balance. Higher = better compression but slower.
	// Level 1: Fastest, ~2x compression
	// Level 3: Default, ~2.5x compression
	// Level 9: High, ~3x compression
	// Level 22: Ultra, ~3.5x compression (very slow)
	CompressionLevel int

	// AutoCheckpointOps triggers automatic checkpoint after N committed operations.
	// Set to 0 to disable operation-based checkpoints.
	// Recommended: 10000 for frequent checkpoints, 100000 for bulk workloads.
	AutoCheckpointOps int

	// AutoCheckpointMB triggers automatic checkpoint when WAL exceeds N megabytes.
	// Set to 0 to disable size-based checkpoints.
	// Recommended: 100 (100MB) for typical workloads, 1000 (1GB) for high-throughput.
	AutoCheckpointMB int

	// DurabilityMode controls fsync behavior (Async, GroupCommit, Sync).
	// Default is DurabilityGroupCommit for balanced performance/durability.
	DurabilityMode DurabilityMode

	// GroupCommitInterval is the maximum time to wait before fsync in GroupCommit mode.
	// Shorter intervals provide better durability but lower throughput.
	// Default: 10ms (100 fsync/sec max)
	GroupCommitInterval time.Duration

	// GroupCommitMaxOps is the maximum operations to batch before fsync in GroupCommit mode.
	// Higher values increase throughput but increase potential data loss on crash.
	// Default: 100 ops
	GroupCommitMaxOps int
}

// DefaultOptions returns default WAL options.
var DefaultOptions = Options{
	Path:                ".",
	Compress:            false,
	CompressionLevel:    3,                     // zstd default level
	AutoCheckpointOps:   10000,                 // Checkpoint every 10k operations
	AutoCheckpointMB:    100,                   // Checkpoint at 100MB WAL size
	DurabilityMode:      DurabilityGroupCommit, // Balanced performance/durability
	GroupCommitInterval: 10 * time.Millisecond, // 100 fsync/sec max
	GroupCommitMaxOps:   100,                   // Batch up to 100 ops
}
