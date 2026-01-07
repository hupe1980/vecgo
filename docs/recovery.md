# Vecgo Crash Recovery Architecture

## Overview

Vecgo uses a **Write-Ahead Log (WAL)** and **Immutable Segments** to ensure durability (ACID).
The recovery process runs automatically on `vector.Open()`.

## Durability Hierarchy

1.  **WAL**: Append-only log of all operations (Insert/Delete). Synced to disk based on configuration (`FsyncEvery`).
2.  **MemTable**: In-memory mutable structure. Reconstructed from WAL on startup if not flushed.
3.  **L0 Segment**: First on-disk immutable structure. Created by Flushing MemTable.
4.  **Tombstones**: Bitmaps tracking deleted rows in immutable segments.

## Crash Scenarios & Recovery

### 1. Crash during Insert (WAL Append)
*   **State**: Partial bytes written to `wal_N.log`.
*   **Recovery**: 
    1.  WAL Reader detects unexpected EOF or checksum mismatch.
    2.  Truncates WAL at last valid record.
    3.  Partial record is discarded (not acknowledged to user, so no data loss vs contract).

### 2. Crash during Flush
*   **State**: Temporary segment file `segment_123.bin.tmp` exists. Manifest points to older generation. WAL is full.
*   **Recovery**:
    1.  `Open` loads Manifest.
    2.  Scans directory for `segment_*.bin`.
    3.  Identifies files NOT in Manifest (orphans).
    4.  Deletes `*.tmp` and orphan `.bin` files.
    5.  Replays WAL from the checkpoint recorded in Manifest.
    6.  Restores MemTable.

### 3. Crash during Compaction
*   **State**: `segment_merged.bin` might exist. Inputs `segment_A.bin` and `segment_B.bin` still exist. Manifest points to A and B.
*   **Recovery**: 
    1.  Loads Manifest (points to A, B).
    2.  Detects `segment_merged.bin` as orphan.
    3.  Deletes orphan.
    4.  System starts with A and B intact. Compaction will be re-scheduled.

### 4. Crash during Manifest Update
*   **State**: `manifest.json.tmp` fully written. Rename to `manifest.json` failed/interrupted.
*   **Recovery**:
    1.  FS guarantees atomic rename (POSIX).
    2.  If `manifest.json` exists, use it.
    3.  If only old manifest exists, use it (new state effectively rolled back).
    4.  If `manifest.json` is corrupt (partial write without tmp?), `ErrIncompatibleFormat` or `ErrCorrupt`. Vecgo uses `manifest.current` pointer file or atomic rename to avoid this.

## Invariants

*   **Idempotency**: WAL replay produces identical MemTable state.
*   **Atomicity**: Segment visibility toggles instantly via Manifest update.
*   **No Data Loss**: Any ack'd write (if Sync=true) is in WAL or Segment.
