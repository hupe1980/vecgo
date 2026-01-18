// Copyright 2024 The Vecgo Authors
// SPDX-License-Identifier: MIT

//go:build (!arm64 && !amd64) || noasm

package vectorstore

import "github.com/hupe1980/vecgo/model"

// prefetchVectorBatch is the portable fallback implementation.
// It uses a volatile read to trigger the hardware prefetcher.
// This works on all architectures but is less efficient than native prefetch.
func prefetchVectorBatch(data []float32, dim, dataLen int, ids []model.RowID) {
	for _, id := range ids {
		idx := int(id) * dim
		if idx >= 0 && idx+dim <= dataLen {
			// Volatile read to trigger hardware prefetcher
			// The compiler cannot optimize this away due to the read
			_ = data[idx]
		}
	}
}
