//go:build !noasm && arm64

package simd

import "unsafe"

//go:noescape
func cosineBatchSve2(query unsafe.Pointer, targets unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

//go:noescape
func dotBatchSve2(query unsafe.Pointer, targets unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

//go:noescape
func squaredL2BatchSve2(query unsafe.Pointer, targets unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)
