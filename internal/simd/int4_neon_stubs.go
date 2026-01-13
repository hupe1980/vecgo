//go:build !noasm && arm64

package simd

import "unsafe"

//go:noescape
func int4L2DistanceBatchNeon(query unsafe.Pointer, codes unsafe.Pointer, dim int64, n int64, min unsafe.Pointer, diff unsafe.Pointer, out unsafe.Pointer)

//go:noescape
func int4L2DistanceNeon(query unsafe.Pointer, code unsafe.Pointer, dim int64, min unsafe.Pointer, diff unsafe.Pointer, out unsafe.Pointer)

//go:noescape
func int4L2DistancePrecomputedNeon(query unsafe.Pointer, code unsafe.Pointer, dim int64, lookupTable unsafe.Pointer, out unsafe.Pointer)

