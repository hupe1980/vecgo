// Package manifest provides Bloom filters for fast categorical negative lookups.
//
// A Bloom filter is a space-efficient probabilistic data structure that can tell us
// definitively if an element is NOT in a set, but may have false positives when
// saying an element IS in the set.
//
// For segment pruning, this is perfect:
//   - If Bloom says "NOT present" → skip segment (100% correct)
//   - If Bloom says "maybe present" → must check segment (may be false positive)
//
// This enables O(1) segment pruning for high-cardinality categorical fields
// where tracking all values (like in TopK) is impractical.
package manifest

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

// ErrCorruptedBloomFilter indicates the Bloom filter data is invalid.
var ErrCorruptedBloomFilter = errors.New("manifest: corrupted bloom filter data")

// BloomFilter is a space-efficient probabilistic filter for categorical values.
// It can definitively say "not in set" but may have false positives for "in set".
//
// Key properties:
//   - False positive rate: ~1% with 10 bits/element, ~0.1% with 14 bits/element
//   - Zero false negatives: if Bloom says "no", it's definitely not there
//   - O(k) lookup where k = number of hash functions (typically 3-7)
//   - Memory: ~1.5 bytes per value for 1% FPR
type BloomFilter struct {
	bits    []uint64 // Bit array (words)
	numBits uint64   // Total bits (for modulo)
	k       uint32   // Number of hash functions
	count   uint32   // Number of elements added
}

// BloomFilterSize computes optimal bloom filter size for given parameters.
// Returns (numBits, numHashFunctions).
//
// For 1% false positive rate: 10 bits/element, k=7
// For 0.1% false positive rate: 14 bits/element, k=10
func BloomFilterSize(expectedElements int, falsePositiveRate float64) (numBits uint64, k uint32) {
	if expectedElements <= 0 {
		expectedElements = 1
	}
	if falsePositiveRate <= 0 || falsePositiveRate >= 1 {
		falsePositiveRate = 0.01 // Default 1%
	}

	// Optimal number of bits: m = -n*ln(p) / (ln(2)^2)
	ln2Sq := math.Ln2 * math.Ln2
	m := float64(-expectedElements) * math.Log(falsePositiveRate) / ln2Sq

	// Optimal number of hash functions: k = (m/n) * ln(2)
	kFloat := (m / float64(expectedElements)) * math.Ln2

	// Round up bits to multiple of 64 for word alignment
	numBits = ((uint64(m) + 63) / 64) * 64
	if numBits < 64 {
		numBits = 64 // Minimum size
	}

	k = uint32(math.Ceil(kFloat))
	if k < 1 {
		k = 1
	}
	if k > 16 {
		k = 16 // Cap to prevent excessive hashing
	}

	return numBits, k
}

// NewBloomFilter creates a new Bloom filter with the specified size and hash count.
func NewBloomFilter(numBits uint64, k uint32) *BloomFilter {
	if numBits < 64 {
		numBits = 64
	}
	// Ensure word alignment
	numBits = ((numBits + 63) / 64) * 64
	if k < 1 {
		k = 1
	}
	if k > 16 {
		k = 16
	}

	return &BloomFilter{
		bits:    make([]uint64, numBits/64),
		numBits: numBits,
		k:       k,
		count:   0,
	}
}

// NewBloomFilterForSize creates a Bloom filter optimized for the expected element count
// with approximately 1% false positive rate.
func NewBloomFilterForSize(expectedElements int) *BloomFilter {
	numBits, k := BloomFilterSize(expectedElements, 0.01)
	return NewBloomFilter(numBits, k)
}

// Add inserts a value into the Bloom filter.
// After Add(x), MayContain(x) will always return true.
func (bf *BloomFilter) Add(value string) {
	h1, h2 := bloomHash(value)
	for i := uint32(0); i < bf.k; i++ {
		// Double hashing: h(i) = h1 + i*h2
		bit := (h1 + uint64(i)*h2) % bf.numBits
		wordIdx := bit / 64
		bitIdx := bit % 64
		bf.bits[wordIdx] |= (1 << bitIdx)
	}
	bf.count++
}

// MayContain checks if a value might be in the filter.
// Returns false: definitely NOT in set (can prune)
// Returns true: maybe in set (cannot prune, must check)
func (bf *BloomFilter) MayContain(value string) bool {
	h1, h2 := bloomHash(value)
	for i := uint32(0); i < bf.k; i++ {
		bit := (h1 + uint64(i)*h2) % bf.numBits
		wordIdx := bit / 64
		bitIdx := bit % 64
		if (bf.bits[wordIdx] & (1 << bitIdx)) == 0 {
			return false // Definitely not in set
		}
	}
	return true // Maybe in set
}

// Count returns the number of elements added to the filter.
func (bf *BloomFilter) Count() uint32 {
	return bf.count
}

// EstimatedFalsePositiveRate returns the estimated false positive rate
// based on the current fill ratio.
func (bf *BloomFilter) EstimatedFalsePositiveRate() float64 {
	if bf.count == 0 {
		return 0
	}
	// FPR ≈ (1 - e^(-k*n/m))^k
	kn := float64(bf.k) * float64(bf.count)
	m := float64(bf.numBits)
	return math.Pow(1-math.Exp(-kn/m), float64(bf.k))
}

// SizeBytes returns the memory size of the filter in bytes.
func (bf *BloomFilter) SizeBytes() int {
	return len(bf.bits) * 8
}

// Clear resets the filter to empty state.
func (bf *BloomFilter) Clear() {
	for i := range bf.bits {
		bf.bits[i] = 0
	}
	bf.count = 0
}

// WriteTo serializes the Bloom filter to a writer.
func (bf *BloomFilter) WriteTo(w io.Writer) (int64, error) {
	var written int64

	// Header: numBits (8) + k (4) + count (4) = 16 bytes
	header := make([]byte, 16)
	binary.LittleEndian.PutUint64(header[0:8], bf.numBits)
	binary.LittleEndian.PutUint32(header[8:12], bf.k)
	binary.LittleEndian.PutUint32(header[12:16], bf.count)

	n, err := w.Write(header)
	written += int64(n)
	if err != nil {
		return written, err
	}

	// Bits array
	for _, word := range bf.bits {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], word)
		n, err := w.Write(buf[:])
		written += int64(n)
		if err != nil {
			return written, err
		}
	}

	return written, nil
}

// ReadBloomFilter deserializes a Bloom filter from a reader.
func ReadBloomFilter(r io.Reader) (*BloomFilter, error) {
	// Read header
	header := make([]byte, 16)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, err
	}

	numBits := binary.LittleEndian.Uint64(header[0:8])
	k := binary.LittleEndian.Uint32(header[8:12])
	count := binary.LittleEndian.Uint32(header[12:16])

	// Validate
	if numBits < 64 || numBits%64 != 0 {
		return nil, ErrCorruptedBloomFilter
	}
	if k < 1 || k > 16 {
		return nil, ErrCorruptedBloomFilter
	}

	// Read bits
	numWords := numBits / 64
	bits := make([]uint64, numWords)
	for i := range bits {
		var buf [8]byte
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			return nil, err
		}
		bits[i] = binary.LittleEndian.Uint64(buf[:])
	}

	return &BloomFilter{
		bits:    bits,
		numBits: numBits,
		k:       k,
		count:   count,
	}, nil
}

// bloomHash computes two independent hash values for double hashing.
// Uses FNV-1a variant for speed and good distribution.
func bloomHash(s string) (h1, h2 uint64) {
	// FNV-1a 64-bit
	const (
		fnvOffset = 14695981039346656037
		fnvPrime  = 1099511628211
	)

	h1 = fnvOffset
	for i := 0; i < len(s); i++ {
		h1 ^= uint64(s[i])
		h1 *= fnvPrime
	}

	// Second hash: different seed and reversed iteration
	h2 = fnvOffset ^ 0x5555555555555555
	for i := len(s) - 1; i >= 0; i-- {
		h2 ^= uint64(s[i])
		h2 *= fnvPrime
	}

	// Ensure h2 is odd (for better double hashing)
	h2 |= 1

	return h1, h2
}

// BloomStats tracks Bloom filter effectiveness.
type BloomStats struct {
	Queries        uint64  // Total queries
	DefiniteNos    uint64  // Bloom said "definitely not" (true negatives)
	MaybeYes       uint64  // Bloom said "maybe yes" (potential false positives)
	ConfirmedFPs   uint64  // Confirmed false positives (after full check)
	ObservedFPRate float64 // Actual observed false positive rate
}

// Update updates the stats with a query result.
func (bs *BloomStats) Update(bloomResult, actualResult bool) {
	bs.Queries++
	if !bloomResult {
		bs.DefiniteNos++ // True negative
	} else {
		bs.MaybeYes++
		if !actualResult {
			bs.ConfirmedFPs++ // False positive
		}
	}
	if bs.MaybeYes > 0 {
		bs.ObservedFPRate = float64(bs.ConfirmedFPs) / float64(bs.MaybeYes)
	}
}

// Effectiveness returns the percentage of queries that were definitively answered.
func (bs *BloomStats) Effectiveness() float64 {
	if bs.Queries == 0 {
		return 0
	}
	return float64(bs.DefiniteNos) / float64(bs.Queries) * 100
}
