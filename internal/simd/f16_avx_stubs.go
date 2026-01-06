//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func f16ToF32Avx(in unsafe.Pointer, out unsafe.Pointer, n int64)

