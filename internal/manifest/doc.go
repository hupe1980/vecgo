// Package manifest implements atomic manifest persistence for the vector database.
//
// # Overview
//
// The manifest is a snapshot of the engine's state at a specific point in time,
// containing segment information, configuration (dimension, metric), and LSN for
// WAL recovery. Manifests enable time-travel queries and crash recovery.
//
// # Binary Format
//
// Manifests are stored in a compact binary format with integrity checking:
//
//	Header (16 bytes):
//	  Magic    (4 bytes) - 0x56454347 ("VECG")
//	  Version  (4 bytes) - Format version (currently 1)
//	  Checksum (4 bytes) - CRC32-IEEE of payload
//	  Length   (4 bytes) - Payload length in bytes
//
//	Payload:
//	  ID            (8 bytes) - Manifest version ID
//	  CreatedAt     (8 bytes) - Unix nanoseconds
//	  Dim           (8 bytes) - Vector dimension
//	  Metric        (string)  - Distance metric name
//	  NextSegmentID (8 bytes) - Next segment ID to allocate
//	  MaxLSN        (8 bytes) - Maximum flushed LSN (for WAL recovery)
//	  NumSegments   (4 bytes) - Number of segments
//	  Segments[]             - Segment metadata
//	  PKIndex.Path  (string)  - Primary key index file path
//
// Strings are length-prefixed (2-byte length + bytes).
//
// # Atomic Protocol
//
// Save follows a two-phase commit protocol for atomic updates:
//
//  1. Write manifest blob to MANIFEST-NNNNNN.bin (where N is the version ID)
//  2. Atomically update CURRENT pointer file to reference the new manifest
//
// On local filesystems, step 2 uses atomic rename. On S3, the strong read-after-write
// consistency guarantee ensures the update is immediately visible.
//
// Load reads CURRENT to find the active manifest filename, then loads that file.
//
// # Thread Safety
//
// All Store methods (Load, LoadVersion, Save, ListVersions, DeleteVersion) are
// protected by a mutex and safe for concurrent use.
//
// # Time Travel
//
// ListVersions returns all available manifest versions, enabling queries against
// historical database states. LoadVersion loads a specific version by ID.
package manifest
