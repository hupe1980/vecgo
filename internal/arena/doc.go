// Package arena provides an off-heap memory allocator for HNSW graphs.
//
// The arena allocator eliminates GC pressure by using mmap-backed memory.
// It provides stable addresses for graph nodes, enabling lock-free concurrent access.
//
// # Features
//
//   - Off-heap allocation via mmap (no GC pressure)
//   - 4MB chunk size for cache locality
//   - Generation tracking for safe memory reclamation
//   - ~4.2ns per allocation
//
// # Safety
//
// All methods return errors instead of panicking. The Get method returns nil
// for invalid offsets rather than panicking.
package arena
