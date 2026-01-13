//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func int4L2DistanceAvx(query unsafe.Pointer, code unsafe.Pointer, dim int64, min unsafe.Pointer, diff unsafe.Pointer, out unsafe.Pointer)

//go:noescape
func int4L2DistanceBatchAvx(query unsafe.Pointer, codes unsafe.Pointer, dim int64, n int64, min unsafe.Pointer, diff unsafe.Pointer, out unsafe.Pointer)

//go:noescape
func int4L2DistancePrecomputedAvx(query unsafe.Pointer, code unsafe.Pointer, dim int64, lookupTable unsafe.Pointer, out unsafe.Pointer)

