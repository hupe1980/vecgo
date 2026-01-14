// Package pk provides a lock-free MVCC primary key index.
//
// The PK index maps auto-increment IDs to physical locations (SegmentID, RowID).
// It supports:
//
//   - Wait-free reads via atomic.Pointer
//   - Lock-free writes via CAS loops
//   - MVCC for snapshot isolation
//   - Time-travel queries
//
// # Architecture
//
// The index uses a paged array structure:
//
//	Index
//	├── atomic.Pointer[pages] ────► []*page (growable slice)
//	│   ├── page[0] ────────────► entries[65536]entry
//	│   │   └── entry.head ─────► atomic.Pointer[version]
//	│   │       └── version ────► {lsn, location, deleted, next}
//	│   ├── page[1]
//	│   └── ...
//	├── count atomic.Int64
//	└── mu sync.RWMutex (for page slice growth only)
//
// IDs are decomposed into (pageIndex, offset) using bit operations:
//
//	pageIndex = id >> 16  (65536 entries per page)
//	offset    = id & 0xFFFF
//
// # Concurrency Model
//
// Reads are wait-free: load atomic page pointer, then load entry head.
// No locks are acquired on the read path.
//
// Writes use CAS loops on the entry's version chain. The page slice
// is protected by RWMutex only during growth operations. A sync.Pool
// recycles version structs to minimize allocations during CAS retries.
//
// Version chains are ordered by LSN (highest first). MVCC queries
// traverse the chain to find the latest version visible to a snapshot.
//
// # Persistence
//
// Save() serializes only the HEAD version of each entry (not full history).
// Load() restores the index from a checkpoint. The binary format uses:
//
//	Magic: 0x504B4958 ("PKIX")
//	Version: 1
//
// # Usage
//
//	idx := pk.New()
//	idx.Upsert(id, location, lsn)        // Insert or update
//	loc, ok := idx.Get(id, snapshotLSN)  // MVCC read
//	idx.Delete(id, lsn)                  // Tombstone
//	idx.Scan(snapshotLSN)                // Iterator
package pk
