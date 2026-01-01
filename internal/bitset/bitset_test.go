package bitset

import (
	"bytes"
	"testing"
)

func TestBitSet(t *testing.T) {
	b := New(100)

	if b.Len() != 100 {
		t.Errorf("expected len 100, got %d", b.Len())
	}

	b.Set(10)
	if !b.Test(10) {
		t.Errorf("expected bit 10 to be set")
	}

	if b.Count() != 1 {
		t.Errorf("expected count 1, got %d", b.Count())
	}

	b.Unset(10)
	if b.Test(10) {
		t.Errorf("expected bit 10 to be unset")
	}

	b.Set(10)
	b.Set(20)
	b.Set(30)

	if b.Count() != 3 {
		t.Errorf("expected count 3, got %d", b.Count())
	}

	b.ClearAll()
	if b.Count() != 0 {
		t.Errorf("expected count 0 after clear, got %d", b.Count())
	}
}

func TestBitSet_Grow(t *testing.T) {
	b := New(10)
	b.Set(5)

	b.Grow(100000) // Should trigger segment growth
	if !b.Test(5) {
		t.Errorf("expected bit 5 to persist after grow")
	}

	b.Set(99999)
	if !b.Test(99999) {
		t.Errorf("expected bit 99999 to be set")
	}
}

func TestBitSet_Serialization(t *testing.T) {
	b := New(1000)
	b.Set(1)
	b.Set(500)
	b.Set(999)

	var buf bytes.Buffer
	_, err := b.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	b2 := New(0)
	_, err = b2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom failed: %v", err)
	}

	if b2.Len() != 1000 {
		t.Errorf("expected len 1000, got %d", b2.Len())
	}
	if !b2.Test(1) || !b2.Test(500) || !b2.Test(999) {
		t.Errorf("serialization lost bits")
	}
}

func TestBitSet_TestAndSet(t *testing.T) {
	b := New(100)
	if b.TestAndSet(10) {
		t.Errorf("expected TestAndSet(10) to return false (was unset)")
	}
	if !b.Test(10) {
		t.Errorf("expected bit 10 to be set")
	}
	if !b.TestAndSet(10) {
		t.Errorf("expected TestAndSet(10) to return true (was set)")
	}
}

func TestBitSet_NextSetBit(t *testing.T) {
	b := New(1000)
	b.Set(10)
	b.Set(20)
	b.Set(100)

	tests := []struct {
		start    uint32
		expected uint32
		found    bool
	}{
		{0, 10, true},
		{10, 10, true},
		{11, 20, true},
		{20, 20, true},
		{21, 100, true},
		{100, 100, true},
		{101, 0, false},
	}

	for _, tt := range tests {
		got, found := b.NextSetBit(tt.start)
		if found != tt.found {
			t.Errorf("NextSetBit(%d) found = %v, expected %v", tt.start, found, tt.found)
		}
		if found && got != tt.expected {
			t.Errorf("NextSetBit(%d) = %d, expected %d", tt.start, got, tt.expected)
		}
	}
}
