package engine

import "errors"

// ErrNotFound is returned when a Store cannot find an ID.
//
// This is an engine-layer sentinel used internally; the vecgo package may
// translate it into its public error contract.
var ErrNotFound = errors.New("not found")

// ErrCoordinatorClosed is returned when an operation is attempted on a closed coordinator.
// This typically happens when trying to submit work to a shutdown worker pool or
// access a shard after Close() has been called.
var ErrCoordinatorClosed = errors.New("coordinator closed")
