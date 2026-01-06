# Vecgo Operations Guide

This guide provides runbooks, monitoring thresholds, and capacity planning for running Vecgo in production.

## 🚨 Failure Runbooks

### Disk Full
**Symptoms**: `Insert` returns `ErrBackpressure` or IO errors; logs show "no space left on device".
**Action**:
1. **Stop Writes**: Immediately stop ingestion to prevent WAL corruption (though Vecgo is crash-safe).
2. **Free Space**: Delete old logs or expand volume.
3. **Restart**: Vecgo will recover from the WAL.
4. **Mitigation**: Configure `Compaction` to run more frequently or use larger disks (1.5x data size).

### OOM (Out of Memory)
**Symptoms**: Process crash (OOM Killer); `ErrBackpressure` on memory limits.
**Action**:
1. **Check Limits**: Ensure `MemoryBudget` in config < Container Memory Limit.
2. **Reduce Budget**: Lower `MemoryBudget` to leave room for OS page cache.
3. **Add Swap**: (Optional) To prevent hard crashes, though performance will degrade.

### Slow Queries (High Latency)
**Symptoms**: p99 latency > 100ms.
**Action**:
1. **Check HNSW Params**: `efSearch` might be too high. Lower it (trade-off: slightly lower recall).
2. **Enable Quantization**: If not enabled, switch to SQ8 or Binary quantization to reduce memory bandwidth.
3. **Check Resource Contention**: CPU saturation? Disk IOPS limit reached?

### Corruption Detected
**Symptoms**: `ErrCorrupt` returned on Open or Search; logs show checksum mismatch.
**Action**:
1. **Restore from Backup**: If WAL is corrupted, restore the last good snapshot.
2. **Rebuild Index**: If a segment is corrupt, you may need to re-ingest data.
3. **File Issue**: Report to maintainers with `docs/development.md` reproduction steps.

## 📊 Monitoring & Alerts

Expose these metrics via `MetricsObserver` (e.g., Prometheus).

| Metric | Threshold | Alert Severity | Description |
|--------|-----------|----------------|-------------|
| `vecgo_backpressure_events_total` | > 10 / min | **Warning** | System is rejecting writes due to overload. |
| `vecgo_compaction_queue_depth` | > 10 | **Warning** | Compaction is falling behind ingestion. |
| `vecgo_search_latency_p99` | > 100ms | **Warning** | Search performance degrading. |
| `vecgo_wal_errors_total` | > 0 | **Critical** | Disk IO failures on write path. |

## 📐 Capacity Planning

### Memory Sizing
**Formula**: `RAM = (Vectors * Dim * 4 bytes) * Overhead + Cache`

- **Raw Vectors**: 1M vectors * 1536 dim * 4 bytes ≈ 6GB
- **Overhead**: HNSW graph links (~2x raw size for M=32)
- **Quantization**: Reduces raw size by 4x (SQ8) or 32x (Binary).

**Recommendation**: Provision **2x working set** RAM for optimal performance (allows OS page cache for segments).

### Disk Sizing
**Formula**: `Disk = Raw Data * 1.5`

- **1.5x Multiplier**: Needed for compaction temporary space.
- **WAL**: Fixed size (e.g., 1GB) + active MemTable dump.

### CPU Sizing
- **Search**: CPU bound. ~2 cores per 1k QPS (highly dependent on `k` and dimensions).
- **Ingest**: IO/CPU bound (hashing/quantization).
