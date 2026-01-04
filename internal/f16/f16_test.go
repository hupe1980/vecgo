package f16

import (
	"math"
	"testing"
)

func TestToFloat32_KnownValues(t *testing.T) {
	tests := []struct {
		name string
		in   Bits
		want float32
	}{
		{"+0", 0x0000, 0},
		{"-0", 0x8000, float32(math.Copysign(0, -1))},
		{"+1", 0x3C00, 1},
		{"-1", 0xBC00, -1},
		{"+Inf", 0x7C00, float32(math.Inf(1))},
		{"-Inf", 0xFC00, float32(math.Inf(-1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ToFloat32(tt.in)
			if tt.name == "-0" {
				if math.Float32bits(got) != math.Float32bits(tt.want) {
					t.Fatalf("got bits=%08x want=%08x", math.Float32bits(got), math.Float32bits(tt.want))
				}
				return
			}
			if got != tt.want {
				t.Fatalf("got=%v want=%v", got, tt.want)
			}
		})
	}
}

func TestToFloat32_SubnormalMin(t *testing.T) {
	// Smallest positive subnormal: 2^-24.
	got := ToFloat32(0x0001)
	want := float32(math.Ldexp(1, -24))
	if got != want {
		t.Fatalf("got=%g want=%g", got, want)
	}
}

func TestToFloat32_NaN(t *testing.T) {
	got := ToFloat32(0x7E00) // canonical quiet NaN in binary16
	if !math.IsNaN(float64(got)) {
		t.Fatalf("expected NaN, got=%v", got)
	}
}

func TestFromFloat32_ZeroSigns(t *testing.T) {
	if got := FromFloat32(0); got != 0x0000 {
		t.Fatalf("+0 got=%04x", uint16(got))
	}
	if got := FromFloat32(float32(math.Copysign(0, -1))); got != 0x8000 {
		t.Fatalf("-0 got=%04x", uint16(got))
	}
}

func TestFromFloat32_InfNaN(t *testing.T) {
	if got := FromFloat32(float32(math.Inf(1))); got != 0x7C00 {
		t.Fatalf("+inf got=%04x", uint16(got))
	}
	if got := FromFloat32(float32(math.Inf(-1))); got != 0xFC00 {
		t.Fatalf("-inf got=%04x", uint16(got))
	}

	nan := float32(math.NaN())
	got := FromFloat32(nan)
	if (got&expMask) != expMask || (got&fracMask) == 0 {
		t.Fatalf("nan encoding not NaN: %04x", uint16(got))
	}
}

func TestFromFloat32_RoundTrip_PowersOfTwo(t *testing.T) {
	// Powers of two within the normal exponent range are exactly representable.
	for e := -14; e <= 15; e++ {
		f := float32(math.Ldexp(1, e))
		h := FromFloat32(f)
		g := ToFloat32(h)
		if g != f {
			t.Fatalf("e=%d f=%g h=%04x g=%g", e, f, uint16(h), g)
		}
	}
}

func TestFromFloat32_RoundingTiesToEven(t *testing.T) {
	// Around 1.0 in binary16: step = 2^-10.
	base := float32(1.0)
	step := float32(math.Ldexp(1, -10))

	// Halfway between 1.0 (even mantissa) and next representable; tie -> even.
	half := base + step/2
	if got := FromFloat32(half); got != 0x3C00 {
		t.Fatalf("halfway up from 1.0 should round to 1.0, got=%04x", uint16(got))
	}

	// Halfway between (1.0+step) and (1.0+2*step). Lower is odd mantissa -> rounds up.
	half2 := base + step + step/2
	if got := FromFloat32(half2); got != 0x3C02 {
		t.Fatalf("halfway with odd lower should round up, got=%04x", uint16(got))
	}
}

func TestEncodeDecode_Slices(t *testing.T) {
	src := []float32{0, 1, -2, 65504, float32(math.Inf(1))}
	h := make([]Bits, len(src))
	Encode(h, src)

	got := make([]float32, len(src))
	Decode(got, h)

	if got[0] != 0 || got[1] != 1 || got[2] != -2 {
		t.Fatalf("unexpected: %v", got)
	}
	if got[3] != 65504 {
		t.Fatalf("max finite got=%v", got[3])
	}
	if !math.IsInf(float64(got[4]), 1) {
		t.Fatalf("inf got=%v", got[4])
	}
}
