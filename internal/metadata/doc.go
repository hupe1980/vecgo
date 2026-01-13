// Package imetadata provides internal metadata indexing for efficient filtering.
//
// This package implements the inverted index used for metadata filtering.
// It is separate from the public metadata package to avoid import cycles.
//
// # Features
//
//   - Roaring Bitmap-based inverted index
//   - String interning for memory efficiency
//   - 3-tier filter routing (bitmap, hybrid, brute-force)
package imetadata
