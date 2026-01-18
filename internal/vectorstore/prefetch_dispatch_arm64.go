// Copyright 2024 The Vecgo Authors
// SPDX-License-Identifier: MIT

//go:build !noasm && arm64

package vectorstore

import (
	"unsafe"

	"github.com/hupe1980/vecgo/model"
)

// prefetchVectorBatch uses ARM64 PRFM instruction for efficient prefetching.
// This is significantly faster than volatile reads as it doesn't stall the pipeline.
// Zero allocation for batches <= 32 elements (common HNSW neighbor count).
func prefetchVectorBatch(data []float32, dim, dataLen int, ids []model.RowID) {
	if len(ids) == 0 {
		return
	}

	// Stack-allocated buffer for common case (HNSW M0 = 32 neighbors)
	var stackBuf [32]uint32
	var idsU32 []uint32

	if len(ids) <= 32 {
		idsU32 = stackBuf[:0]
	} else {
		// Rare case: more than 32 neighbors
		idsU32 = make([]uint32, 0, len(ids))
	}

	// Filter valid IDs
	for _, id := range ids {
		idx := int(id) * dim
		if idx >= 0 && idx+dim <= dataLen {
			idsU32 = append(idsU32, uint32(id))
		}
	}

	if len(idsU32) > 0 {
		prefetchBatchNEON(unsafe.Pointer(&data[0]), dim, len(idsU32), idsU32)
	}
}
