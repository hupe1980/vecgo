//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func dotProductAvx(a unsafe.Pointer, b unsafe.Pointer, n int64, res unsafe.Pointer)

//go:noescape
func pqAdcLookupAvx(table unsafe.Pointer, codes unsafe.Pointer, m int64, result unsafe.Pointer, offsets_ptr unsafe.Pointer)

//go:noescape
func squaredL2Avx(vec1 unsafe.Pointer, vec2 unsafe.Pointer, n int64, result unsafe.Pointer)

