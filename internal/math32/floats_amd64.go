//go:build amd64 && !noasm

package math32

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	useAVX = cpu.X86.HasAVX
	useAVX512 = cpu.X86.HasAVX512
}

//go:noescape
func _dot_product_avx(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func _dot_product_avx512(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func _squared_l2_avx(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func _squared_l2_avx512(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func _pq_adc_lookup_avx(table, codes unsafe.Pointer, m int64, result unsafe.Pointer)

//go:noescape
func _pq_adc_lookup_avx512(table, codes unsafe.Pointer, m int64, result unsafe.Pointer)

func dot(a, b []float32) float32 {
	switch {
	case useAVX512:
		var ret float32

		if len(a) > 0 {
			_dot_product_avx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
		}

		return ret
	case useAVX:
		var ret float32

		if len(a) > 0 {
			_dot_product_avx(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
		}

		return ret
	default:
		return dotGeneric(a, b)
	}
}

func squaredL2(a, b []float32) float32 {
	switch {
	case useAVX512:
		var ret float32

		if len(a) > 0 {
			_squared_l2_avx512(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
		}

		return ret
	case useAVX:
		var ret float32

		if len(a) > 0 {
			_squared_l2_avx(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
		}

		return ret
	default:
		return squaredL2Generic(a, b)
	}
}

func scaleInPlace(a []float32, scalar float32) {
	scaleGeneric(a, scalar)
}

func pqAdcLookup(table []float32, codes []byte, m int) float32 {
	switch {
	case useAVX512:
		var ret float32
		if m > 0 {
			_pq_adc_lookup_avx512(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
		}
		return ret
	case useAVX:
		var ret float32
		if m > 0 {
			_pq_adc_lookup_avx(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
		}
		return ret
	default:
		return pqAdcLookupGeneric(table, codes, m)
	}
}
