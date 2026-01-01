package engine

import (
	"iter"

	"github.com/hupe1980/vecgo/core"
)

// Store is a generic interface for storing and retrieving data associated with vector IDs.
//
// This previously lived in the top-level `store` package. It was moved into `engine`
// to avoid a separate public package while keeping Vecgo/engine decoupled.
//
// Implementations can provide different storage strategies (in-memory, disk-backed, distributed, etc.).
type Store[T any] interface {
	// Get retrieves the data associated with the given ID.
	// Returns the data and true if found, or zero value and false if not found.
	Get(id core.LocalID) (T, bool)

	// Set stores data associated with the given ID.
	// If the ID already exists, it updates the data.
	Set(id core.LocalID, data T) error
	// Delete removes the data associated with the given ID.
	// Returns an error if the ID doesn't exist.
	Delete(id core.LocalID) error

	// BatchGet retrieves data for multiple IDs in a single operation.
	// Returns a map of id -> data for all found IDs.
	BatchGet(ids []core.LocalID) (map[core.LocalID]T, error)
	// BatchSet stores multiple id -> data pairs in a single operation.
	// If any operation fails, the entire batch may be rolled back (implementation-dependent).
	BatchSet(items map[core.LocalID]T) error

	// BatchDelete removes data for multiple IDs in a single operation.
	BatchDelete(ids []core.LocalID) error

	// Len returns the number of items currently stored.
	Len() int

	// Clear removes all items from the store.
	Clear() error

	// ToMap returns a copy of all data as a map (for serialization).
	ToMap() map[core.LocalID]T

	// All returns an iterator over all items in the store.
	All() iter.Seq2[core.LocalID, T]
}
