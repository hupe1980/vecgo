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
// # Implementation
//
// Uses a paged array structure for O(1) access with sequential IDs.
// No hashing overhead compared to map-based implementations.
package pk
