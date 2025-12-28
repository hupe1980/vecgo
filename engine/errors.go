package engine

import "errors"

// ErrNotFound is returned when a Store cannot find an ID.
//
// This is an engine-layer sentinel used internally; the vecgo package may
// translate it into its public error contract.
var ErrNotFound = errors.New("not found")
