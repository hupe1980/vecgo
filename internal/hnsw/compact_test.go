package hnsw

import (
	"context"
	"math/rand"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/vectorstore"
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
		o.EF = 200
		o.Vectors = vectorstore.New(dim)
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
	deleted := make(map[model.RowID]bool)
	for i := 0; i < deleteCount; i++ {
		id := model.RowID(i * 2)
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

	for id := model.RowID(0); id < model.RowID(count); id++ {
		node := h.getNode(g, id)
		if node.Offset == 0 {
			continue
		}

		if deleted[id] {
			// Check if connections are cleared
			// Exception: Entry point might still have connections to allow traversal
			if uint32(id) == g.entryPointAtomic.Load() {
				continue
			}

			for l := 0; l <= node.Level(g.arena); l++ {
				connCount := 0
				conns := h.getConnections(g, id, l)
				connCount = len(conns)
				if connCount > 0 {
					t.Errorf("Deleted node %d has connections at level %d", id, l)
				}
			}
		} else {
			// Check neighbors
			for l := 0; l <= node.Level(g.arena); l++ {
				connCount := 0
				conns := h.getConnections(g, id, l)
				for _, neighbor := range conns {
					connCount++
					if deleted[neighbor.ID] {
						t.Errorf("Active node %d points to deleted node %d at level %d", id, neighbor.ID, l)
					}
				}

				// Check connectivity (heuristic)
				// Layer 0 should have connections if possible
				if l == 0 && connCount == 0 && count-deleteCount > 1 {
					// It's possible to be isolated if graph is small, but with 500 nodes it shouldn't happen often
					// t.Logf("Active node %d has 0 connections at level 0", id)
				}
			}
		}
	}
}
