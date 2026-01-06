//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func dotBatchAvx512(query unsafe.Pointer, targets unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

//go:noescape
func squaredL2BatchAvx512(query unsafe.Pointer, targets unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

