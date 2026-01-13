// Package model defines core types used throughout Vecgo.
//
// # Identity Types
//
//   - ID: Globally unique, auto-incrementing primary key (uint64)
//   - SegmentID: Unique identifier for a segment (uint64)
//   - RowID: Segment-local record identifier (uint32)
//   - Location: Physical address (SegmentID, RowID, Version)
//
// # Data Types
//
//   - Record: Vector with optional metadata and payload
//   - Candidate: Search result with ID, score, and optional data
//   - SearchOptions: Configuration for search queries
//
// # Record Builder
//
// Use the fluent API to construct records:
//
//	rec := model.NewRecord(vec).
//	    WithMetadata("category", metadata.String("tech")).
//	    WithPayload(jsonData).
//	    Build()
package model
