//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func sq8L2BatchAvx512(query unsafe.Pointer, codes unsafe.Pointer, scales unsafe.Pointer, biases unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

