package distance

import "testing"

func TestSquaredL2(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0}
	v2 := []float32{4.0, 5.0, 6.0}

	result := SquaredL2(v1, v2)

	// Expected: (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
	expected := float32(27.0)
	if result != expected {
		t.Errorf("SquaredL2 distance = %v, want %v", result, expected)
	}
}

func TestDot(t *testing.T) {
	v1 := []float32{1.0, 2.0, 3.0}
	v2 := []float32{4.0, 5.0, 6.0}

	result := Dot(v1, v2)

	// Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	expected := float32(32.0)
	if result != expected {
		t.Errorf("Dot product = %v, want %v", result, expected)
	}
}
