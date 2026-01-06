//go:build !noasm && arm64

package simd

import "unsafe"

//go:noescape
func sq8L2BatchNeon(query unsafe.Pointer, codes unsafe.Pointer, scales unsafe.Pointer, biases unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

//go:noescape
func sq8uL2BatchPerDimensionNeon(query unsafe.Pointer, codes unsafe.Pointer, mins unsafe.Pointer, invScales unsafe.Pointer, dim int64, n int64, out unsafe.Pointer)

