# WAL format (current)

This document describes the current on-disk format used by the `wal` package.

## File header

The WAL starts with a self-describing header:

- Magic: `VGW0` (4 bytes)
- Version: uint16 little-endian
- Flags: uint16 little-endian
  - bit 0: compressed (zstd)
- Compression level: uint8 (only meaningful when compressed)
- Reserved: 3 bytes
- Codec name length: uint16 little-endian
- Reserved: 2 bytes
- Codec name bytes (UTF-8)

The codec name is required and is used to reject opening a WAL with a different **metadata codec**.

## Entry stream

After the header, the file contains an entry stream.

Entries are encoded by `(*WAL).encodeEntry` and decoded by `(*WAL).decodeEntry`.
The stream may be compressed with zstd if enabled.

A checkpoint is represented as an `OpCheckpoint` entry; replay stops at the first checkpoint.

## Recovery semantics

- On-disk entries use the Prepare/Commit protocol only:
  - `OpPrepareInsert` + `OpCommitInsert`
  - `OpPrepareUpdate` + `OpCommitUpdate`
  - `OpPrepareDelete` + `OpCommitDelete`
- `ReplayCommitted` applies only operations that have a matching commit.
- `Replay` iterates the raw on-disk entries (including prepare/commit markers).
