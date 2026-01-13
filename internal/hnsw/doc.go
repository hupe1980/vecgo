// Package hnsw implements Hierarchical Navigable Small World graphs.
//
// HNSW provides approximate nearest neighbor search with high recall and
// sub-linear query time. This implementation is optimized for:
//
// # Features
//
//   - 16-way sharding for write concurrency
//   - Lock-free search path
//   - Arena-based allocation (no GC pressure)
//   - Dynamic EF (ACORN-lite) for filtered search
//   - Lock-free RNG (xorshift64*)
//
// # Parameters
//
//   - M: Max connections per node (default: 32)
//   - EF: Construction queue size (default: 300)
//   - EFSearch: Search queue size (default: max(k+100, 200))
//
// # Reference
//
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
// using Hierarchical Navigable Small World graphs", IEEE TPAMI 2018.
package hnsw
