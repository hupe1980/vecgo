//go:build arm64 && !noasm

package math32

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.ARM64.HasASIMD {
		dotImpl = dotNEON
		squaredL2Impl = squaredL2NEON
		scaleImpl = scaleNEON
		pqAdcImpl = pqAdcNEON
	}
}

//go:noescape
func _dot_product_neon(a unsafe.Pointer, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func _squared_l2_neon(a, b unsafe.Pointer, n int64, result unsafe.Pointer)

//go:noescape
func _pq_adc_lookup_neon(table, codes unsafe.Pointer, m int64, result unsafe.Pointer)

// NOTE: The generated assembly currently expects 3 args laid out as:
// a @ +0, n @ +8, scalar @ +16.
//
//go:noescape
func _scale_neon(a unsafe.Pointer, n int64, scalar unsafe.Pointer)

func dotNEON(a, b []float32) float32 {
	var ret float32

	if len(a) > 0 {
		_dot_product_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}

	return ret
}

func squaredL2NEON(a, b []float32) float32 {
	var ret float32

	if len(a) > 0 {
		_squared_l2_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), int64(len(a)), unsafe.Pointer(&ret))
	}

	return ret
}

func scaleNEON(a []float32, scalar float32) {
	if len(a) == 0 {
		return
	}
	s := scalar
	_scale_neon(unsafe.Pointer(&a[0]), int64(len(a)), unsafe.Pointer(&s))
}

func pqAdcNEON(table []float32, codes []byte, m int) float32 {
	var ret float32
	if m > 0 {
		_pq_adc_lookup_neon(unsafe.Pointer(&table[0]), unsafe.Pointer(&codes[0]), int64(m), unsafe.Pointer(&ret))
	}
	return ret
}
