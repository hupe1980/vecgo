// Copyright 2024 The Vecgo Authors
// SPDX-License-Identifier: MIT

//go:build !noasm && arm64

package vectorstore

import "unsafe"

//go:noescape
func prefetchVectorNEON(ptr unsafe.Pointer)

//go:noescape
func prefetchBatchNEON(base unsafe.Pointer, dim, count int, ids []uint32)
