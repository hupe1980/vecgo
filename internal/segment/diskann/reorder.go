package diskann

import (
	"context"
	"fmt"

	imetadata "github.com/hupe1980/vecgo/internal/metadata"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
)

// reorderBFS reorders the graph and all associated data in BFS order
// to improve cache locality during search.
func (w *Writer) reorderBFS(ctx context.Context) error {
	n := len(w.vectors)
	if n == 0 {
		return nil
	}

	// 1. Compute BFS Order
	// perm[newID] = oldID
	perm := make([]uint32, 0, n)
	// invPerm[oldID] = newID
	invPerm := make([]uint32, n)
	// Initialize invPerm with -1 to detect unvisited
	for i := range invPerm {
		invPerm[i] = ^uint32(0) // MaxUint32
	}

	visited := make([]bool, n)
	queue := make([]uint32, 0, 1024)

	// Start BFS from entry point
	queue = append(queue, w.entryPoint)
	visited[w.entryPoint] = true

	// Standard BFS
	head := 0
	for head < len(queue) {
		curr := queue[head]
		head++

		// Record mapping
		newID := uint32(len(perm))
		perm = append(perm, curr)
		invPerm[curr] = newID

		// Add neighbors to queue
		for _, neighbor := range w.graph[curr] {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}

	// Handle disconnected components (if any)
	// Iterate through all nodes to find unvisited ones
	for i := 0; i < n; i++ {
		if !visited[i] {
			// Start BFS from this unvisited node
			queue = append(queue, uint32(i))
			visited[i] = true

			for head < len(queue) {
				curr := queue[head]
				head++

				newID := uint32(len(perm))
				perm = append(perm, curr)
				invPerm[curr] = newID

				for _, neighbor := range w.graph[curr] {
					if !visited[neighbor] {
						visited[neighbor] = true
						queue = append(queue, neighbor)
					}
				}
			}
		}
	}

	if len(perm) != n {
		return fmt.Errorf("permutation length mismatch: expected %d, got %d", n, len(perm))
	}

	// 2. Apply Permutation to Data
	newVectors := make([][]float32, n)
	newIds := make([]model.ID, n)
	newMetadata := make([][]byte, n)
	newPayloads := make([][]byte, n)
	newGraph := make([][]uint32, n)

	var newCompressedVectors [][]byte
	if len(w.compressedVectors) > 0 {
		newCompressedVectors = make([][]byte, n)
	}

	var newBQCodes [][]uint64
	if len(w.bqCodes) > 0 {
		newBQCodes = make([][]uint64, n)
	}

	for newID, oldID := range perm {
		newVectors[newID] = w.vectors[oldID]
		newIds[newID] = w.ids[oldID]
		newMetadata[newID] = w.metadata[oldID]
		newPayloads[newID] = w.payloads[oldID]

		// Remap edges
		oldEdges := w.graph[oldID]
		newEdges := make([]uint32, len(oldEdges))
		for i, neighborOldID := range oldEdges {
			newEdges[i] = invPerm[neighborOldID]
		}
		newGraph[newID] = newEdges

		if newCompressedVectors != nil {
			newCompressedVectors[newID] = w.compressedVectors[oldID]
		}
		if newBQCodes != nil {
			newBQCodes[newID] = w.bqCodes[oldID]
		}
	}

	// 3. Update Writer state
	w.vectors = newVectors
	w.ids = newIds
	w.metadata = newMetadata
	w.payloads = newPayloads
	w.graph = newGraph
	w.compressedVectors = newCompressedVectors
	w.bqCodes = newBQCodes

	// Remap Entry Point
	w.entryPoint = invPerm[w.entryPoint]

	// 4. Rebuild Inverted Index
	// Since RowIDs changed, the old inverted index is invalid.
	w.index = imetadata.NewUnifiedIndex()

	for i, data := range w.metadata {
		if len(data) > 0 {
			var md metadata.Document
			if err := md.UnmarshalBinary(data); err != nil {
				return fmt.Errorf("failed to unmarshal metadata during reorder at new RowID %d: %w", i, err)
			}
			w.index.AddInvertedIndex(model.RowID(i), md)
		}
	}

	// Store the permutation for callers to retrieve final RowID mapping.
	// invPerm[oldAddOrderIndex] = finalRowID
	w.addOrderToFinalRow = invPerm

	return nil
}
