package hnsw

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/hupe1980/vecgo/vectorstore/columnar"
)

func TestCompact(t *testing.T) {
	ctx := context.Background()
	dim := 16
	count := 1000
	deleteCount := 500

	// Create index
	h, err := New(func(o *Options) {
		o.Dimension = dim
		o.M = 16
		o.EF = 100
		o.Vectors = columnar.New(dim)
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Insert vectors
	rng := rand.New(rand.NewSource(42))
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rng.Float32()
		}
		if _, err := h.Insert(ctx, vectors[i]); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}

	// Delete vectors (every second one)
	deleted := make(map[uint64]bool)
	for i := 0; i < deleteCount; i++ {
		id := uint64(i * 2)
		if err := h.Delete(ctx, id); err != nil {
			t.Fatalf("Delete: %v", err)
		}
		deleted[id] = true
	}

	// Run Compact
	if err := h.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	// Verify graph integrity
	// 1. Deleted nodes should have no connections (we cleared them)
	// 2. Active nodes should not point to deleted nodes
	// 3. Active nodes should have reasonable number of connections (repaired)

	g := h.currentGraph.Load()

	for id := uint64(0); id < uint64(count); id++ {
		node := h.getNode(g, id)
		if node == nil {
			continue
		}

		if deleted[id] {
			// Check if connections are cleared
			// Exception: Entry point might still have connections to allow traversal
			if id == g.entryPointAtomic.Load() {
				continue
			}

			for l := 0; l <= node.Level; l++ {
				conns := node.getConnections(l)
				if len(conns) > 0 {
					t.Errorf("Deleted node %d has connections at level %d", id, l)
				}
			}
		} else {
			// Check neighbors
			for l := 0; l <= node.Level; l++ {
				conns := node.getConnections(l)
				for _, neighbor := range conns {
					if deleted[neighbor.ID] {
						t.Errorf("Active node %d points to deleted node %d at level %d", id, neighbor.ID, l)
					}
				}

				// Check connectivity (heuristic)
				// Layer 0 should have connections if possible
				if l == 0 && len(conns) == 0 && count-deleteCount > 1 {
					// It's possible to be isolated if graph is small, but with 500 nodes it shouldn't be.
					// Unless it's the only node.
					t.Logf("Warning: Active node %d has 0 connections at level 0", id)
				}
			}
		}
	}

	// Verify Recall
	// Search for some active vectors
	hits := 0
	queries := 100
	for i := 0; i < queries; i++ {
		// Pick an active vector as query
		targetID := uint64((i * 2) + 1) // Odd IDs are active
		if targetID >= uint64(count) {
			break
		}
		query := vectors[targetID]

		results, err := h.KNNSearch(ctx, query, 10, nil)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}

		// Check if targetID is in results
		found := false
		for _, r := range results {
			if r.ID == targetID {
				found = true
				break
			}
		}
		if found {
			hits++
		}
	}

	precision := float64(hits) / float64(queries)
	fmt.Printf("Precision after compaction: %f\n", precision)
	if precision < 0.9 {
		t.Errorf("Precision too low: %f", precision)
	}
}
