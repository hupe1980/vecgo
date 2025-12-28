//go:build arm64 && !noasm

package math32

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func init() {
	useNEON = cpu.ARM64.HasASIMD
}

//go:noescape
func _dot_product_neon(a unsafe.Pointer, b unsafe.Pointer, n uintptr, ret unsafe.Pointer)

//go:noescape
func _squared_l2_neon(a, b unsafe.Pointer, n uintptr, result unsafe.Pointer)

// NOTE: The generated assembly currently expects 4 args laid out as:
// a @ +0, n @ +8, scalarPtr @ +16, unused @ +24.
//
//go:noescape
func _scale_neon(a unsafe.Pointer, n uintptr, scalarPtr unsafe.Pointer, unused unsafe.Pointer)

func dot(a, b []float32) float32 {
	switch {
	case useNEON:
		var ret float32

		if len(a) > 0 {
			_dot_product_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), uintptr(len(a)), unsafe.Pointer(&ret))
		}

		return ret
	default:
		return dotGeneric(a, b)
	}
}

func squaredL2(a, b []float32) float32 {
	switch {
	case useNEON:
		var ret float32

		if len(a) > 0 {
			_squared_l2_neon(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), uintptr(len(a)), unsafe.Pointer(&ret))
		}

		return ret
	default:
		return squaredL2Generic(a, b)
	}
}

func scaleInPlace(a []float32, scalar float32) {
	switch {
	case useNEON:
		if len(a) == 0 {
			return
		}
		s := scalar
		_scale_neon(unsafe.Pointer(&a[0]), uintptr(len(a)), unsafe.Pointer(&s), nil)
	default:
		scaleGeneric(a, scalar)
	}
}
