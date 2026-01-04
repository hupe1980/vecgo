package engine

import "github.com/hupe1980/vecgo/internal/wal"

// Durability controls the durability guarantees of the WAL.
//
// This is part of Vecgo's public API surface (via vecgo re-exports).
// The underlying WAL implementation is internal.
type Durability int

const (
	// DurabilityAsync relies on OS page cache. Fast but risky.
	DurabilityAsync Durability = iota
	// DurabilitySync calls fsync after every write. Slow but safe.
	DurabilitySync
)

// WALOptions configures the write-ahead log.
type WALOptions struct {
	Durability Durability
}

func DefaultWALOptions() WALOptions {
	return WALOptions{Durability: DurabilitySync}
}

func toInternalWALOptions(o WALOptions) wal.Options {
	// Defensive mapping: default to Sync for unknown values.
	d := o.Durability
	if d != DurabilityAsync && d != DurabilitySync {
		d = DurabilitySync
	}
	return wal.Options{Durability: wal.Durability(d)}
}
