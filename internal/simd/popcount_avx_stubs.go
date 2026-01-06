//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func hammingAvx(a unsafe.Pointer, b unsafe.Pointer, n int64, lookup_ptr unsafe.Pointer, low_mask_ptr unsafe.Pointer) int64

//go:noescape
func popcountAvx(a unsafe.Pointer, n int64, lookup_ptr unsafe.Pointer, low_mask_ptr unsafe.Pointer) int64

