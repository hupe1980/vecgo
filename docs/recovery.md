# Vecgo Crash Recovery & Durability

Vecgo guarantees **ACID** properties for single-shard operations and **crash safety** for all durable artifacts. This document details the recovery algorithm.

## Durability Architecture

### 1. Write-Ahead Log (WAL)
- **Format**: Append-only sequence of `Insert/Delete` records.
- **Sync Mode**: Configurable (`Sync` = fsync every write, `Async` = fsync every N ms).
- **Checksums**: CRC32C per record.

### 2. Immutable Segments (LSM Tree)
- **State**: Once written, a segment file (`.seg`) is immutable.
- **Publication**: Atomic `rename()` from temp file to final filename.

### 3. Manifest
- **Role**: Source of truth for the current set of active segments.
- **Update**: Atomic `rename()` of new manifest file.

## Recovery Algorithm

On `Open()`, Vecgo performs the following steps:

### Step 1: Load Manifest
1. Find the latest valid `manifest-{seq}.json` file.
2. Validate checksum.
3. Load list of active segments.
4. **Cleanup**: Delete any `.seg` files not referenced in the manifest (orphaned compaction outputs).

### Step 2: Replay WAL
1. Identify the WAL file associated with the active MemTable.
2. Read records sequentially.
3. **Checksum Check**: If a record has a mismatch (partial write at crash):
   - **Truncate**: Discard the corrupted record and all subsequent bytes.
   - **Log**: Emit a warning.
4. **Re-Apply**: Insert valid records into the MemTable.
   - **Idempotency**: Operations are idempotent, so replaying is safe even if partially applied before crash.

### Step 3: Consistency Check
1. Verify all segments in manifest exist and have valid headers.
2. If a segment is missing/corrupt -> Return `ErrCorrupt` (requires manual intervention/backup restore).

## Crash Scenarios & Guarantees

| Scenario | State at Restart | Recovery Action | Data Loss? |
|----------|------------------|-----------------|------------|
| **Crash during WAL append** | Partial record at end of WAL | Truncate WAL at last valid record | Last write lost (Async) / No loss (Sync) |
| **Crash during Flush** | Temp segment exists, Manifest old | Ignore/Delete temp segment, Replay WAL | None |
| **Crash during Compaction** | Temp merged segment exists | Ignore/Delete temp segment | None |
| **Crash during Manifest Update** | New manifest partial/missing | Load old manifest, treat new segments as orphans | None |
| **Power Loss** | Data in OS page cache | Depends on `fsync` policy | Last N ms (Async) / None (Sync) |

## Invariants

1. **Atomicity**: A record is either fully visible or fully missing.
2. **Monotonicity**: If a write is acknowledged (and fsynced), it will be present after recovery.
3. **No Resurrection**: Deleted keys remain deleted after compaction/recovery.
