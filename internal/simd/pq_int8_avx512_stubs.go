//go:build !noasm && amd64

package simd

import "unsafe"

//go:noescape
func buildDistanceTableInt8Avx512(query unsafe.Pointer, codebook unsafe.Pointer, subdim int64, scale unsafe.Pointer, offset unsafe.Pointer, out unsafe.Pointer)

//go:noescape
func findNearestCentroidInt8Avx512(query unsafe.Pointer, codebook unsafe.Pointer, subdim int64, scale unsafe.Pointer, offset unsafe.Pointer, outIndex unsafe.Pointer)

//go:noescape
func squaredL2Int8DequantizedAvx512(query unsafe.Pointer, code unsafe.Pointer, subdim int64, scale unsafe.Pointer, offset unsafe.Pointer, out unsafe.Pointer)

