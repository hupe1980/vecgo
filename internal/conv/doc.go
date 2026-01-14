// Package conv provides safe integer type conversion utilities.
//
// These functions perform bounds checking to prevent integer overflow/underflow
// when converting between signed/unsigned and different bit-width integer types.
//
// Use cases:
//   - Validating untrusted data from disk (file headers, counts, offsets)
//   - Converting between Go's int (platform-dependent) and fixed-width types
//
// For conversions that are provably safe by domain constraints (e.g., loop
// indices, bounded counters), use direct type casts instead to avoid overhead.
package conv
