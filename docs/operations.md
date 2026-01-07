# Vecgo Operations Runbook

This guide covers operational procedures for managing Vecgo in production.

## Monitoring

Key metrics to alert on (via Prometheus/MetricsObserver):

| Metric | Threshold | Investigation |
|--------|-----------|---------------|
| `vecgo_backpressure_events_total` | > 10 / min | System is overloaded. Increase memory limits or shard writes. |
| `vecgo_compaction_queue_depth` | > 10 | Disk I/O bottleneck. Check disk IOPS or tune compaction policy. |
| `vecgo_operation_latency_seconds{op="search", type="l0"}` | p99 > 100ms | CPU contention or unoptimized HNSW params. |
| `vecgo_memtable_size_bytes` | > 90% of Limit | High write pressure. Check flush configuration. |

## Failure Scenarios

### Disk Full

**Symptoms**:
- `Insert` returns `ErrBackpressure` or IO errors.
- Logs show "write: no space left on device".

**Resolution**:
1. **Immediate**: Add disk space or delete old logs/files.
2. **Recovery**: Vecgo handles disk-full gracefully. Once space is available, writes will succeed.
   - If flush failed, MemTable remains in memory.
   - If WAL append failed, partial record is truncated on restart.

**Prevention**:
- Set `MaxWALSize` to trigger flushes earlier.
- Monitor disk usage alerts at 80%.

### Out of Memory (OOM)

**Symptoms**:
- Process crash (kernel OOM killer).
- `ErrBackpressure` returned if `ResourceController` is effective.

**Resolution**:
1. Check `vecgo_memtable_size_bytes` and `vecgo_cache_size_bytes`.
2. Tune `WithResourceController` limits to be lower than container limit.
3. Reduce `WithBlockCacheSize` (default 256MB).

### Corruption Detected

**Symptoms**:
- `ErrCorrupt` returned on Open.
- Logs indicate "checksum mismatch" or "invalid magic".

**Resolution**:
1. **WAL Corruption**: Vecgo truncates corrupted tail entries automatically (if configured). Othwerise, manual truncation may be needed.
2. **Segment Corruption**: Delete the corrupted `.bin` file. Vecgo will load remaining segments (might lose data in that segment).
3. **Restore**: If manifest is corrupt, wipe directory and restore from backup.

## Capacity Planning

**Formula**:
```
RAM = (MemTableSize) + (BlockCacheSize) + (IndexOverhead)
Disk = (RawVectorSize * 1.5) + (WALSize)
```

**Example (10M vectors, 1536 dim, float32)**:
- Raw Data: 10M * 1536 * 4B = ~60GB
- Disk: 60GB * 1.5 (compaction) = 90GB
- RAM: 
  - BlockCache: 4GB (recommended)
  - MemTable: 1GB
  - HNSW (L0): ~5% of L0 vectors (if keeping L0 small)
  - **Total**: ~8GB RAM machine recommended.
