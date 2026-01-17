package manifest

import (
	"bytes"
	"testing"
)

func TestBloomFilter_Basic(t *testing.T) {
	bf := NewBloomFilterForSize(1000)

	// Add some values
	values := []string{"apple", "banana", "cherry", "date", "elderberry"}
	for _, v := range values {
		bf.Add(v)
	}

	// Check all added values are "maybe present"
	for _, v := range values {
		if !bf.MayContain(v) {
			t.Errorf("MayContain(%q) = false, want true (no false negatives allowed)", v)
		}
	}

	// Check count
	if bf.Count() != uint32(len(values)) {
		t.Errorf("Count() = %d, want %d", bf.Count(), len(values))
	}
}

func TestBloomFilter_DefinitelyNot(t *testing.T) {
	bf := NewBloomFilterForSize(100)

	// Add some values
	bf.Add("foo")
	bf.Add("bar")
	bf.Add("baz")

	// These should have very high probability of returning false
	// (definitively not in set) since filter is mostly empty
	notPresent := []string{
		"definitely_not_here_12345",
		"another_missing_value_67890",
		"xyz_missing_abc",
	}

	falsePositives := 0
	for _, v := range notPresent {
		if bf.MayContain(v) {
			falsePositives++
		}
	}

	// With 100 element filter and only 3 elements added,
	// false positive rate should be very low
	if falsePositives > 1 {
		t.Logf("Got %d false positives out of %d (may happen occasionally)", falsePositives, len(notPresent))
	}
}

func TestBloomFilter_FalsePositiveRate(t *testing.T) {
	// Test with known parameters
	n := 10000 // elements
	bf := NewBloomFilterForSize(n)

	// Add n elements
	for i := 0; i < n; i++ {
		bf.Add(string(rune('A' + (i % 26))))
	}

	// Estimated FPR should be reasonable
	fpr := bf.EstimatedFalsePositiveRate()
	if fpr > 0.5 {
		t.Errorf("EstimatedFalsePositiveRate() = %f, want < 0.5", fpr)
	}

	t.Logf("Estimated FPR after %d insertions: %.4f", n, fpr)
}

func TestBloomFilter_Serialization(t *testing.T) {
	// Create and populate a filter
	bf := NewBloomFilter(512, 7)
	values := []string{"one", "two", "three", "four", "five"}
	for _, v := range values {
		bf.Add(v)
	}

	// Serialize
	var buf bytes.Buffer
	n, err := bf.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}
	if n != int64(buf.Len()) {
		t.Errorf("WriteTo returned %d, but buffer has %d bytes", n, buf.Len())
	}

	// Deserialize
	bf2, err := ReadBloomFilter(&buf)
	if err != nil {
		t.Fatalf("ReadBloomFilter failed: %v", err)
	}

	// Verify parameters match
	if bf2.numBits != bf.numBits {
		t.Errorf("numBits mismatch: got %d, want %d", bf2.numBits, bf.numBits)
	}
	if bf2.k != bf.k {
		t.Errorf("k mismatch: got %d, want %d", bf2.k, bf.k)
	}
	if bf2.count != bf.count {
		t.Errorf("count mismatch: got %d, want %d", bf2.count, bf.count)
	}

	// Verify all values still "may contain"
	for _, v := range values {
		if !bf2.MayContain(v) {
			t.Errorf("After deserialization, MayContain(%q) = false", v)
		}
	}
}

func TestBloomFilterSize(t *testing.T) {
	tests := []struct {
		n   int
		fpr float64
	}{
		{1000, 0.01},  // 1% FPR
		{1000, 0.001}, // 0.1% FPR
		{10000, 0.01},
		{100000, 0.01},
	}

	for _, tc := range tests {
		numBits, k := BloomFilterSize(tc.n, tc.fpr)
		bitsPerElement := float64(numBits) / float64(tc.n)
		t.Logf("n=%d, FPR=%.3f → %d bits (%.1f bits/elem), k=%d",
			tc.n, tc.fpr, numBits, bitsPerElement, k)

		// Sanity checks
		if numBits < 64 {
			t.Errorf("numBits too small: %d", numBits)
		}
		if k < 1 || k > 16 {
			t.Errorf("k out of range: %d", k)
		}
	}
}

func TestBloomStats(t *testing.T) {
	bs := &BloomStats{}

	// Simulate some queries
	bs.Update(false, false) // True negative
	bs.Update(false, false) // True negative
	bs.Update(true, true)   // True positive
	bs.Update(true, false)  // False positive

	if bs.Queries != 4 {
		t.Errorf("Queries = %d, want 4", bs.Queries)
	}
	if bs.DefiniteNos != 2 {
		t.Errorf("DefiniteNos = %d, want 2", bs.DefiniteNos)
	}
	if bs.MaybeYes != 2 {
		t.Errorf("MaybeYes = %d, want 2", bs.MaybeYes)
	}
	if bs.ConfirmedFPs != 1 {
		t.Errorf("ConfirmedFPs = %d, want 1", bs.ConfirmedFPs)
	}

	eff := bs.Effectiveness()
	if eff != 50.0 {
		t.Errorf("Effectiveness = %f, want 50.0", eff)
	}
}

func BenchmarkBloomFilter_Add(b *testing.B) {
	bf := NewBloomFilterForSize(b.N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bf.Add("value_" + string(rune('A'+(i%26))))
	}
}

func BenchmarkBloomFilter_MayContain(b *testing.B) {
	bf := NewBloomFilterForSize(10000)
	for i := 0; i < 10000; i++ {
		bf.Add("value_" + string(rune('A'+(i%26))))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bf.MayContain("query_" + string(rune('A'+(i%26))))
	}
}

func BenchmarkBloomFilter_MayContain_Hit(b *testing.B) {
	bf := NewBloomFilterForSize(10000)
	for i := 0; i < 10000; i++ {
		bf.Add("value_" + string(rune('A'+(i%26))))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bf.MayContain("value_A") // Always a hit
	}
}
