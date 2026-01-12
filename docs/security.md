# Vecgo Security & Safety

Vecgo is an **embedded library**. Security is a **shared responsibility** between Vecgo and the embedding application.

## Responsibility Matrix

| Feature | Vecgo Responsibility | Application Responsibility |
|---------|----------------------|----------------------------|
| **Authentication** | N/A | Verify user identity before calling Vecgo. |
| **Authorization** | N/A | Check permissions (e.g., "Can User X search Index Y?"). |
| **Encryption (Rest)**| N/A (Files are plain binary) | Encrypt disk volume (LUKS/EBS) or encrypt payloads before insert. |
| **Encryption (Transit)**| N/A (In-process) | TLS for any network API exposing Vecgo. |
| **Input Validation** | Reject invalid vectors/params | Validate user inputs before passing to Vecgo. |
| **DoS Protection** | Resource limits (Backpressure) | Rate limiting, Quotas. |

## Safety Features

### 1. Resource Exhaustion Protection
Vecgo prevents trivial DoS attacks via:
- **Memory Limits**: Hard caps on MemTable and Cache. Returns `ErrBackpressure` if exceeded.
- **Dimension Validation**: Rejects vectors with the wrong dimension for the index (`ErrInvalidArgument`).
- **Batch Behavior**: Very large batches can still exhaust configured budgets and trigger `ErrBackpressure`; there is no fixed “max batch size” enforced by default.

### 2. Corruption Detection
- **Checksums**: CRC32C on WAL records and Segment headers.
- **Safe Decoding**: All binary decoding validates bounds. No panics on malformed data.
- **Fast Fail**: `ErrIncompatibleFormat` for version mismatches.

### 3. Safe Defaults
- **Memory**: Default 1GB limit.
- **WAL**: Sync mode by default (strongest durability).

## Recommendations for Secure Embedding

1. **Sanitize Inputs**: Do not pass untrusted arbitrary strings as filters without validation.
2. **Isolate Tenants**: Use separate Vecgo instances (directories) for different tenants to ensure data isolation.
3. **Monitor Resources**: Watch `vecgo_backpressure_events` to detect DoS attempts.
4. **Update Regularly**: Keep Vecgo updated for security patches.
