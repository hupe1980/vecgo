// Package f16 implements IEEE-754 binary16 (float16) encoding/decoding.

// This package is internal: it exists to support float16 as a storage format
// while keeping execution in float32.
package f16

import (
	"math"
)

// Bits is the raw IEEE-754 binary16 bit-pattern.

// Layout:
//
//	sign: 1 bit
//	exp:  5 bits (bias 15)
//	frac: 10 bits
type Bits uint16

const (
	signMask Bits = 0x8000
	expMask  Bits = 0x7C00
	fracMask Bits = 0x03FF

	f32ExpMask  uint32 = 0x7F800000
	f32FracMask uint32 = 0x007FFFFF
)

// ToFloat32 converts a binary16 bit-pattern to float32.
func ToFloat32(h Bits) float32 {
	sign := uint32(h&signMask) << 16
	exp := uint32(h&expMask) >> 10
	frac := uint32(h & fracMask)

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: normalize the fraction.
		// Half subnormals have an exponent of -14 and no implicit leading 1.
		// We normalize to construct a float32 normal.
		e := int32(-14)
		m := frac
		for (m & 0x0400) == 0 {
			m <<= 1
			e--
		}
		m &= 0x03FF // strip leading 1
		f32Exp := uint32(int32(127)+e) << 23
		f32Frac := m << 13
		return math.Float32frombits(sign | f32Exp | f32Frac)
	case 0x1F:
		// Inf/NaN
		if frac == 0 {
			return math.Float32frombits(sign | f32ExpMask)
		}
		return math.Float32frombits(sign | f32ExpMask | (frac << 13))
	default:
		// Normalized
		f32Exp := uint32(int32(exp)-15+127) << 23
		f32Frac := frac << 13
		return math.Float32frombits(sign | f32Exp | f32Frac)
	}
}

// FromFloat32 converts a float32 value into a binary16 bit-pattern.
//
// Rounding mode: round-to-nearest, ties-to-even.
func FromFloat32(f float32) Bits {
	bits := math.Float32bits(f)
	sign := Bits((bits >> 16) & uint32(signMask))
	exp := int32((bits & f32ExpMask) >> 23)
	frac := bits & f32FracMask

	// NaN / Inf
	if exp == 0xFF {
		if frac == 0 {
			return sign | Bits(expMask) // infinity
		}
		// Preserve some payload; ensure it's a quiet NaN and non-zero.
		payload := Bits(frac >> 13)
		if payload == 0 {
			payload = 1
		}
		payload |= 0x0200
		return sign | Bits(expMask) | (payload & fracMask)
	}

	// Zero (and float32 subnormals underflow to zero for binary16 in practice)
	if exp == 0 {
		return sign
	}

	// Re-bias exponent from float32 (127) to float16 (15).
	e16 := exp - 127 + 15

	// Overflow -> Inf
	if e16 >= 0x1F {
		return sign | Bits(expMask)
	}

	// Underflow -> subnormal/zero
	if e16 <= 0 {
		// Too small even for subnormal.
		if e16 < -10 {
			return sign
		}
		// Make the implicit leading 1 explicit.
		mant := frac | 0x00800000
		// Shift so that we end up with a 10-bit mantissa.
		shift := uint32(1-int32(e16)) + 13
		m := mant >> shift
		remainder := mant & ((uint32(1) << shift) - 1)
		half := uint32(1) << (shift - 1)
		if remainder > half || (remainder == half && (m&1) == 1) {
			m++
		}
		return sign | Bits(m)
	}

	// Normal case: convert fraction (23 bits) -> (10 bits) with rounding.
	m := frac >> 13
	remainder := frac & 0x1FFF
	if remainder > 0x1000 || (remainder == 0x1000 && (m&1) == 1) {
		m++
		if m == 0x0400 {
			// Mantissa overflow; carry into exponent.
			m = 0
			e16++
			if e16 >= 0x1F {
				return sign | Bits(expMask)
			}
		}
	}

	return sign | Bits(uint32(e16)<<10) | Bits(m)
}

// Decode converts a slice of binary16 bit-patterns to float32.
// dst must have length >= len(src).
func Decode(dst []float32, src []Bits) {
	for i := range src {
		dst[i] = ToFloat32(src[i])
	}
}

// Encode converts a slice of float32 to binary16.
// dst must have length >= len(src).
func Encode(dst []Bits, src []float32) {
	for i := range src {
		dst[i] = FromFloat32(src[i])
	}
}
