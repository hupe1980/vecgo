//go:build !noasm && arm64

package simd

import "unsafe"

//go:noescape
func f16ToF32Neon(in unsafe.Pointer, out unsafe.Pointer, n int64)

