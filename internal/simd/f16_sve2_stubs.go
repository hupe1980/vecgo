//go:build !noasm && arm64

package simd

import "unsafe"

//go:noescape
func f16DotProductSve2(a unsafe.Pointer, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func f16SquaredL2Sve2(a unsafe.Pointer, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func f16ToF32Sve2(in unsafe.Pointer, out unsafe.Pointer, n int64)

//go:noescape
func f32ToF16Sve2(in unsafe.Pointer, out unsafe.Pointer, n int64)
