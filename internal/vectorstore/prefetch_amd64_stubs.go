// Copyright 2024 The Vecgo Authors
// SPDX-License-Identifier: MIT

//go:build !noasm && amd64

package vectorstore

import "unsafe"

//go:noescape
func prefetchBatchAMD64(base unsafe.Pointer, dim, count int, ids []uint32)
