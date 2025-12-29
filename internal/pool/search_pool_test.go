package pool

import (
	"sync"
	"testing"
)

func TestSearchContext_Basic(t *testing.T) {
	ctx := Get()
	defer Put(ctx)

	// Test initial state
	if ctx.IsVisited(0) {
		t.Error("New context should have no visited nodes")
	}

	// Test marking visited
	if ctx.MarkVisited(0) {
		t.Error("First visit should return false")
	}

	if !ctx.IsVisited(0) {
		t.Error("Node 0 should be marked as visited")
	}

	if !ctx.MarkVisited(0) {
		t.Error("Second visit should return true")
	}
}

func TestSearchContext_Capacity(t *testing.T) {
	ctx := Get()
	defer Put(ctx)

	// Test growth
	largeID := uint32(200000)
	ctx.EnsureVisitedCapacity(largeID)

	if ctx.maxNodes <= largeID {
		t.Errorf("Capacity should be > %d, got %d", largeID, ctx.maxNodes)
	}

	// Mark nodes across range
	ctx.MarkVisited(0)
	ctx.MarkVisited(100000)
	ctx.MarkVisited(largeID)

	if !ctx.IsVisited(0) || !ctx.IsVisited(100000) || !ctx.IsVisited(largeID) {
		t.Error("All marked nodes should be visited")
	}
}

func TestSearchContext_Reset(t *testing.T) {
	ctx := Get()
	defer Put(ctx)

	// Mark some nodes
	ctx.MarkVisited(0)
	ctx.MarkVisited(100)
	ctx.MarkVisited(1000)

	// Reset
	ctx.Reset()

	// Verify cleared
	if ctx.IsVisited(0) || ctx.IsVisited(100) || ctx.IsVisited(1000) {
		t.Error("Reset should clear all visited nodes")
	}

	stats := ctx.Stats()
	if stats.VisitedCount != 0 {
		t.Errorf("Visited count should be 0 after reset, got %d", stats.VisitedCount)
	}
}

func TestSearchContext_Pool(t *testing.T) {
	// Test that pool reuses contexts
	ctx1 := Get()
	ctx1.MarkVisited(42)
	Put(ctx1)

	ctx2 := Get()
	defer Put(ctx2)

	// Should be same object (reused)
	// But should be reset (no visited nodes)
	if ctx2.IsVisited(42) {
		t.Error("Pooled context should be reset")
	}
}

func TestSearchContext_Concurrent(t *testing.T) {
	const numGoroutines = 100
	const opsPerGoroutine = 1000

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()

			for j := 0; j < opsPerGoroutine; j++ {
				ctx := Get()

				// Do some work
				for k := 0; k < 100; k++ {
					ctx.MarkVisited(uint32(k))
				}

				stats := ctx.Stats()
				if stats.VisitedCount != 100 {
					t.Errorf("Expected 100 visited, got %d", stats.VisitedCount)
				}

				Put(ctx)
			}
		}()
	}

	wg.Wait()
}

func BenchmarkSearchContext_Get(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		ctx := Get()
		Put(ctx)
	}
}

func BenchmarkSearchContext_MarkVisited(b *testing.B) {
	ctx := Get()
	defer Put(ctx)

	b.ResetTimer()
	b.ReportAllocs()
	var i int
	for b.Loop() {
		ctx.MarkVisited(uint32(i % 10000))
		i++
	}
}

func BenchmarkSearchContext_MarkVisited_Map(b *testing.B) {
	// Compare with map[uint32]bool
	visited := make(map[uint32]bool)

	b.ResetTimer()
	b.ReportAllocs()
	var i int
	for b.Loop() {
		visited[uint32(i%10000)] = true
		i++
	}
}

func BenchmarkSearchContext_IsVisited(b *testing.B) {
	ctx := Get()
	defer Put(ctx)

	// Mark half the nodes
	for i := 0; i < 10000; i += 2 {
		ctx.MarkVisited(uint32(i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	var i int
	for b.Loop() {
		_ = ctx.IsVisited(uint32(i % 10000))
		i++
	}
}
