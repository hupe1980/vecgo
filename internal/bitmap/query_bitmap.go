package bitmap

import (
	"math/bits"
	"sync"

	"github.com/hupe1980/vecgo/internal/simd"
)

// BlockSize is the number of uint64 words per block (512 bits = 64 bytes).
// Chosen for cache line alignment and SIMD register width compatibility.
const BlockSize = 8

// WordBits is the number of bits per word.
const WordBits = 64

// BlockBits is the number of bits per block.
const BlockBits = BlockSize * WordBits // 512 bits

// BlocksPerMaskWord is the number of blocks tracked per activeBlocks word.
const BlocksPerMaskWord = 64

// QueryBitmap is a zero-allocation, SIMD-friendly bitmap for query-time filter operations.
//
// Key properties:
//   - Fixed-size universe (set at pool creation)
//   - Monotonic construction (Add, AddRange)
//   - SIMD-accelerated AND/OR/ANDNOT/Cardinality
//   - Cache-line aligned blocks (64 bytes)
//   - Two-level active block tracking: []uint64 mask enables O(1) block skipping
//   - Per-block popcount cache for O(activeBlocks) cardinality
//   - Pooled and reusable (zero allocation in steady state)
//
// Memory layout:
//
//	┌─────────────────────────────────────────────────────────────────────┐
//	│  Block 0 (64B)  │  Block 1 (64B)  │  Block 2 (64B)  │ ...           │
//	│  8 × uint64     │  8 × uint64     │  8 × uint64     │               │
//	│  bits [0,511]   │  bits [512,1023]│  bits [1024,1535│               │
//	└─────────────────────────────────────────────────────────────────────┘
//
// Active block mask ([]uint64):
//
//	┌────────────────────────────────────────────────────────────────┐
//	│  Word 0: blocks 0-63  │  Word 1: blocks 64-127  │ ...          │
//	│  bit i = 1 if block i has any set bits                         │
//	└────────────────────────────────────────────────────────────────┘
//
// This layout enables:
//   - SIMD AND/OR on 512-bit chunks (AVX-512, NEON 4×128)
//   - Hardware prefetch efficiency
//   - No container dispatch overhead (unlike Roaring)
//   - Skip 64 blocks at once via TrailingZeros64 on activeBlocks
//   - O(activeBlocks) cardinality via blockPopcounts
type QueryBitmap struct {
	// words is the backing storage: contiguous 64-bit words.
	// Organized as blocks of 8 words (64 bytes, cache-line aligned).
	words []uint64

	// activeBlocks is a bitset tracking which blocks have at least one bit set.
	// Each bit represents one block (512 bits). Enables O(1) block skipping.
	// This is the key optimization: skip 64 blocks at once via TrailingZeros64.
	activeBlocks []uint64

	// blockPopcounts caches the popcount per block.
	// Enables O(activeBlocks) cardinality calculation.
	// Set to 0xFFFF to indicate needs recalculation.
	blockPopcounts []uint16

	// universeSize is the maximum ID + 1 (determines allocation size).
	universeSize uint32

	// numBlocks is the total number of blocks.
	numBlocks int

	// cardinality is cached. Set to -1 to indicate needs recalculation.
	cardinality int

	// ownedByPool indicates if this bitmap came from a pool.
	ownedByPool bool
}

// New creates a new QueryBitmap with the given universe size.
func New(universeSize uint32) *QueryBitmap {
	numWords := (universeSize + WordBits - 1) / WordBits
	// Round up to block boundary for SIMD alignment
	numWords = ((numWords + BlockSize - 1) / BlockSize) * BlockSize
	numBlocks := int(numWords / BlockSize)
	numMaskWords := (numBlocks + BlocksPerMaskWord - 1) / BlocksPerMaskWord

	return &QueryBitmap{
		words:          make([]uint64, numWords),
		activeBlocks:   make([]uint64, numMaskWords),
		blockPopcounts: make([]uint16, numBlocks),
		universeSize:   universeSize,
		numBlocks:      numBlocks,
		cardinality:    0,
	}
}

// EnsureCapacity grows the bitmap if needed to accommodate IDs up to newSize-1.
// If the current universe is >= newSize, this is a no-op. Zero allocations if capacity is sufficient.
func (qb *QueryBitmap) EnsureCapacity(newSize uint32) {
	if newSize <= qb.universeSize {
		return
	}

	numWords := (newSize + WordBits - 1) / WordBits
	numWords = ((numWords + BlockSize - 1) / BlockSize) * BlockSize
	numBlocks := int(numWords / BlockSize)
	numMaskWords := (numBlocks + BlocksPerMaskWord - 1) / BlocksPerMaskWord

	// Grow words slice
	if int(numWords) > len(qb.words) {
		newWords := make([]uint64, numWords)
		copy(newWords, qb.words)
		qb.words = newWords
	}

	// Grow activeBlocks slice
	if numMaskWords > len(qb.activeBlocks) {
		newActive := make([]uint64, numMaskWords)
		copy(newActive, qb.activeBlocks)
		qb.activeBlocks = newActive
	}

	// Grow blockPopcounts slice
	if numBlocks > len(qb.blockPopcounts) {
		newPopcounts := make([]uint16, numBlocks)
		copy(newPopcounts, qb.blockPopcounts)
		qb.blockPopcounts = newPopcounts
	}

	qb.universeSize = newSize
	qb.numBlocks = numBlocks
}

// setBlockActive sets the active bit for a block.
//
//go:nosplit
func (qb *QueryBitmap) setBlockActive(blockIdx int) {
	maskWord := blockIdx / BlocksPerMaskWord
	maskBit := blockIdx % BlocksPerMaskWord
	qb.activeBlocks[maskWord] |= uint64(1) << maskBit
}

// clearBlockActive clears the active bit for a block.
//
//go:nosplit
func (qb *QueryBitmap) clearBlockActive(blockIdx int) {
	maskWord := blockIdx / BlocksPerMaskWord
	maskBit := blockIdx % BlocksPerMaskWord
	qb.activeBlocks[maskWord] &^= uint64(1) << maskBit
}

// isBlockActive checks if a block is marked active.
//
//go:nosplit
func (qb *QueryBitmap) isBlockActive(blockIdx int) bool {
	maskWord := blockIdx / BlocksPerMaskWord
	maskBit := blockIdx % BlocksPerMaskWord
	return qb.activeBlocks[maskWord]&(uint64(1)<<maskBit) != 0
}

// invalidateBlockPopcount marks a block's popcount for recalculation.
//
//go:nosplit
func (qb *QueryBitmap) invalidateBlockPopcount(blockIdx int) {
	qb.blockPopcounts[blockIdx] = 0xFFFF
}

// Clear resets the bitmap to empty state. Zero allocations.
// Only touches active blocks (huge win for sparse bitmaps).
func (qb *QueryBitmap) Clear() {
	// Iterate active blocks using the bitset - skip 64 blocks at once
	for maskIdx, mask := range qb.activeBlocks {
		for mask != 0 {
			// Get next active block
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit

			// Clear the block's words
			start := blockIdx * BlockSize
			for i := start; i < start+BlockSize; i++ {
				qb.words[i] = 0
			}
			qb.blockPopcounts[blockIdx] = 0

			mask &= mask - 1 // Clear lowest bit
		}
		qb.activeBlocks[maskIdx] = 0
	}
	qb.cardinality = 0
}

// Add sets a single bit. Returns true if the bit was newly set.
func (qb *QueryBitmap) Add(id uint32) bool {
	if id >= qb.universeSize {
		return false
	}

	wordIdx := id / WordBits
	bitPos := id % WordBits
	mask := uint64(1) << bitPos

	if qb.words[wordIdx]&mask != 0 {
		return false // Already set
	}

	qb.words[wordIdx] |= mask
	blockIdx := int(wordIdx / BlockSize)
	qb.setBlockActive(blockIdx)
	qb.invalidateBlockPopcount(blockIdx)

	if qb.cardinality >= 0 {
		qb.cardinality++
	}
	return true
}

// AddUnchecked sets a single bit without bounds checking or cardinality update.
//
//go:nosplit
func (qb *QueryBitmap) AddUnchecked(id uint32) {
	wordIdx := id / WordBits
	bitPos := id % WordBits
	qb.words[wordIdx] |= uint64(1) << bitPos
	blockIdx := int(wordIdx / BlockSize)
	qb.setBlockActive(blockIdx)
	qb.invalidateBlockPopcount(blockIdx)
}

// AddMany sets multiple bits efficiently.
// Optimized for sorted inputs (common case from roaring extraction):
// batches block tracking updates within the same block.
func (qb *QueryBitmap) AddMany(ids []uint32) {
	if len(ids) == 0 {
		return
	}

	lastBlockIdx := -1
	for _, id := range ids {
		if id >= qb.universeSize {
			continue
		}
		wordIdx := id / WordBits
		bitPos := id % WordBits
		qb.words[wordIdx] |= uint64(1) << bitPos

		// Only update block tracking when we move to a new block
		blockIdx := int(wordIdx / BlockSize)
		if blockIdx != lastBlockIdx {
			qb.setBlockActive(blockIdx)
			qb.invalidateBlockPopcount(blockIdx)
			lastBlockIdx = blockIdx
		}
	}
	qb.cardinality = -1
}

// AddRange sets all bits in [start, end). Fast path for numeric filters.
func (qb *QueryBitmap) AddRange(start, end uint32) {
	if start >= end || start >= qb.universeSize {
		return
	}
	if end > qb.universeSize {
		end = qb.universeSize
	}

	startWord := start / WordBits
	endWord := (end - 1) / WordBits
	startBit := start % WordBits
	endBit := (end - 1) % WordBits

	if startWord == endWord {
		mask := (^uint64(0) >> (63 - endBit + startBit)) << startBit
		qb.words[startWord] |= mask
	} else {
		qb.words[startWord] |= ^uint64(0) << startBit
		for w := startWord + 1; w < endWord; w++ {
			qb.words[w] = ^uint64(0)
		}
		qb.words[endWord] |= ^uint64(0) >> (63 - endBit)
	}

	// Mark all touched blocks as active and invalidate popcounts
	startBlock := int(startWord / BlockSize)
	endBlock := int(endWord / BlockSize)
	for b := startBlock; b <= endBlock; b++ {
		qb.setBlockActive(b)
		qb.invalidateBlockPopcount(b)
	}

	qb.cardinality = -1
}

// Contains checks if a bit is set. O(1).
func (qb *QueryBitmap) Contains(id uint32) bool {
	if id >= qb.universeSize {
		return false
	}
	wordIdx := id / WordBits
	bitPos := id % WordBits
	return qb.words[wordIdx]&(uint64(1)<<bitPos) != 0
}

// IsEmpty returns true if no bits are set.
func (qb *QueryBitmap) IsEmpty() bool {
	if qb.cardinality >= 0 {
		return qb.cardinality == 0
	}

	// Check if any active blocks exist
	for _, mask := range qb.activeBlocks {
		if mask != 0 {
			return false
		}
	}
	qb.cardinality = 0
	return true
}

// computeBlockPopcount calculates and caches the popcount for a block.
func (qb *QueryBitmap) computeBlockPopcount(blockIdx int) uint16 {
	if qb.blockPopcounts[blockIdx] != 0xFFFF {
		return qb.blockPopcounts[blockIdx]
	}

	start := blockIdx * BlockSize
	count := 0
	for i := start; i < start+BlockSize; i++ {
		count += bits.OnesCount64(qb.words[i])
	}
	qb.blockPopcounts[blockIdx] = uint16(count)
	return uint16(count)
}

// Cardinality returns the number of set bits.
// O(activeBlocks) using cached block popcounts.
func (qb *QueryBitmap) Cardinality() int {
	if qb.cardinality >= 0 {
		return qb.cardinality
	}

	count := 0
	// Iterate active blocks using bitset - skip 64 blocks at once
	for maskIdx, mask := range qb.activeBlocks {
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			count += int(qb.computeBlockPopcount(blockIdx))
			mask &= mask - 1
		}
	}
	qb.cardinality = count
	return count
}

// Cardinality64 returns the cardinality as uint64 (for segment.Bitmap interface).
func (qb *QueryBitmap) Cardinality64() uint64 {
	return uint64(qb.Cardinality())
}

// InvalidateCardinality marks cardinality for recalculation.
func (qb *QueryBitmap) InvalidateCardinality() {
	qb.cardinality = -1
}

// And performs in-place intersection: qb = qb AND other.
// Uses active-mask-driven execution: skip SIMD entirely for inactive regions.
func (qb *QueryBitmap) And(other *QueryBitmap) {
	numMaskWords := min(len(qb.activeBlocks), len(other.activeBlocks))

	for maskIdx := 0; maskIdx < numMaskWords; maskIdx++ {
		// Compute intersection of active masks - only process blocks active in BOTH
		activeMask := qb.activeBlocks[maskIdx] & other.activeBlocks[maskIdx]
		deadMask := qb.activeBlocks[maskIdx] &^ other.activeBlocks[maskIdx]

		// Clear blocks that are active in qb but not in other (result is 0)
		for deadMask != 0 {
			bit := bits.TrailingZeros64(deadMask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize
			for i := start; i < start+BlockSize; i++ {
				qb.words[i] = 0
			}
			qb.blockPopcounts[blockIdx] = 0
			deadMask &= deadMask - 1
		}

		// AND only the blocks active in both
		for activeMask != 0 {
			bit := bits.TrailingZeros64(activeMask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize

			// SIMD AND the block
			simd.AndWords(qb.words[start:start+BlockSize], other.words[start:start+BlockSize])

			// Check if block is now empty
			blockEmpty := true
			for i := start; i < start+BlockSize; i++ {
				if qb.words[i] != 0 {
					blockEmpty = false
					break
				}
			}
			if blockEmpty {
				qb.clearBlockActive(blockIdx)
				qb.blockPopcounts[blockIdx] = 0
			} else {
				qb.invalidateBlockPopcount(blockIdx)
			}

			activeMask &= activeMask - 1
		}

		// Update the active mask
		qb.activeBlocks[maskIdx] &= other.activeBlocks[maskIdx]
	}

	// Clear blocks beyond other's range
	for maskIdx := numMaskWords; maskIdx < len(qb.activeBlocks); maskIdx++ {
		mask := qb.activeBlocks[maskIdx]
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize
			for i := start; i < start+BlockSize; i++ {
				qb.words[i] = 0
			}
			qb.blockPopcounts[blockIdx] = 0
			mask &= mask - 1
		}
		qb.activeBlocks[maskIdx] = 0
	}

	qb.cardinality = -1
}

// AndNot performs in-place difference: qb = qb AND NOT other.
func (qb *QueryBitmap) AndNot(other *QueryBitmap) {
	numMaskWords := min(len(qb.activeBlocks), len(other.activeBlocks))

	for maskIdx := 0; maskIdx < numMaskWords; maskIdx++ {
		// Only process blocks that are active in qb AND other
		activeMask := qb.activeBlocks[maskIdx] & other.activeBlocks[maskIdx]

		for activeMask != 0 {
			bit := bits.TrailingZeros64(activeMask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize

			simd.AndNotWords(qb.words[start:start+BlockSize], other.words[start:start+BlockSize])

			// Check if block is now empty
			blockEmpty := true
			for i := start; i < start+BlockSize; i++ {
				if qb.words[i] != 0 {
					blockEmpty = false
					break
				}
			}
			if blockEmpty {
				qb.clearBlockActive(blockIdx)
				qb.blockPopcounts[blockIdx] = 0
			} else {
				qb.invalidateBlockPopcount(blockIdx)
			}

			activeMask &= activeMask - 1
		}
	}

	qb.cardinality = -1
}

// Or performs in-place union: qb = qb OR other.
func (qb *QueryBitmap) Or(other *QueryBitmap) {
	numMaskWords := min(len(qb.activeBlocks), len(other.activeBlocks))

	for maskIdx := 0; maskIdx < numMaskWords; maskIdx++ {
		// Only process blocks active in other (they're the ones adding bits)
		newBits := other.activeBlocks[maskIdx] &^ qb.activeBlocks[maskIdx]
		overlap := qb.activeBlocks[maskIdx] & other.activeBlocks[maskIdx]

		// Copy blocks that are new (only in other)
		for newBits != 0 {
			bit := bits.TrailingZeros64(newBits)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize
			copy(qb.words[start:start+BlockSize], other.words[start:start+BlockSize])
			qb.blockPopcounts[blockIdx] = other.blockPopcounts[blockIdx]
			newBits &= newBits - 1
		}

		// OR blocks that overlap
		for overlap != 0 {
			bit := bits.TrailingZeros64(overlap)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize
			simd.OrWords(qb.words[start:start+BlockSize], other.words[start:start+BlockSize])
			qb.invalidateBlockPopcount(blockIdx)
			overlap &= overlap - 1
		}

		// Update active mask
		qb.activeBlocks[maskIdx] |= other.activeBlocks[maskIdx]
	}

	qb.cardinality = -1
}

// Xor performs in-place symmetric difference: qb = qb XOR other.
func (qb *QueryBitmap) Xor(other *QueryBitmap) {
	numMaskWords := min(len(qb.activeBlocks), len(other.activeBlocks))

	for maskIdx := 0; maskIdx < numMaskWords; maskIdx++ {
		// Process all blocks active in either bitmap
		eitherActive := qb.activeBlocks[maskIdx] | other.activeBlocks[maskIdx]

		for eitherActive != 0 {
			bit := bits.TrailingZeros64(eitherActive)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize

			simd.XorWords(qb.words[start:start+BlockSize], other.words[start:start+BlockSize])

			// Check if block is now empty or has bits
			blockActive := false
			for i := start; i < start+BlockSize; i++ {
				if qb.words[i] != 0 {
					blockActive = true
					break
				}
			}
			if blockActive {
				qb.setBlockActive(blockIdx)
			} else {
				qb.clearBlockActive(blockIdx)
			}
			qb.invalidateBlockPopcount(blockIdx)

			eitherActive &= eitherActive - 1
		}
	}

	qb.cardinality = -1
}

// IntersectionCount returns |qb AND other| without modifying either bitmap.
func (qb *QueryBitmap) IntersectionCount(other *QueryBitmap) int {
	count := 0
	numMaskWords := min(len(qb.activeBlocks), len(other.activeBlocks))

	for maskIdx := 0; maskIdx < numMaskWords; maskIdx++ {
		// Only count blocks active in both
		activeMask := qb.activeBlocks[maskIdx] & other.activeBlocks[maskIdx]

		for activeMask != 0 {
			bit := bits.TrailingZeros64(activeMask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize

			for i := start; i < start+BlockSize; i++ {
				count += bits.OnesCount64(qb.words[i] & other.words[i])
			}
			activeMask &= activeMask - 1
		}
	}
	return count
}

// UnionCount returns |qb OR other| without modifying either bitmap.
func (qb *QueryBitmap) UnionCount(other *QueryBitmap) int {
	count := 0
	numMaskWords := min(len(qb.activeBlocks), len(other.activeBlocks))

	for maskIdx := 0; maskIdx < numMaskWords; maskIdx++ {
		onlyQb := qb.activeBlocks[maskIdx] &^ other.activeBlocks[maskIdx]
		onlyOther := other.activeBlocks[maskIdx] &^ qb.activeBlocks[maskIdx]
		both := qb.activeBlocks[maskIdx] & other.activeBlocks[maskIdx]

		// Blocks only in qb
		for onlyQb != 0 {
			bit := bits.TrailingZeros64(onlyQb)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			count += int(qb.computeBlockPopcount(blockIdx))
			onlyQb &= onlyQb - 1
		}

		// Blocks only in other
		for onlyOther != 0 {
			bit := bits.TrailingZeros64(onlyOther)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize
			for i := start; i < start+BlockSize; i++ {
				count += bits.OnesCount64(other.words[i])
			}
			onlyOther &= onlyOther - 1
		}

		// Blocks in both - need to OR
		for both != 0 {
			bit := bits.TrailingZeros64(both)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			start := blockIdx * BlockSize
			for i := start; i < start+BlockSize; i++ {
				count += bits.OnesCount64(qb.words[i] | other.words[i])
			}
			both &= both - 1
		}
	}

	// Count remaining blocks in qb beyond other's range
	for maskIdx := numMaskWords; maskIdx < len(qb.activeBlocks); maskIdx++ {
		mask := qb.activeBlocks[maskIdx]
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit
			count += int(qb.computeBlockPopcount(blockIdx))
			mask &= mask - 1
		}
	}

	return count
}

// Rank returns the number of set bits up to and including index i.
func (qb *QueryBitmap) Rank(i uint32) int {
	if i >= qb.universeSize {
		return qb.Cardinality()
	}

	wordIdx := i / WordBits
	targetBlock := int(wordIdx / BlockSize)
	rank := 0

	// Sum complete blocks before target block using cached popcounts
	for maskIdx := 0; maskIdx*BlocksPerMaskWord <= targetBlock; maskIdx++ {
		mask := qb.activeBlocks[maskIdx]
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit

			if blockIdx < targetBlock {
				rank += int(qb.computeBlockPopcount(blockIdx))
			} else if blockIdx == targetBlock {
				// Partial block - count words before target word
				start := blockIdx * BlockSize
				targetWordInBlock := int(wordIdx) - start
				for w := 0; w < targetWordInBlock; w++ {
					rank += bits.OnesCount64(qb.words[start+w])
				}
				// Partial word
				bitPos := i % WordBits
				mask := (uint64(1) << (bitPos + 1)) - 1
				rank += bits.OnesCount64(qb.words[wordIdx] & mask)
				return rank
			}
			mask &= mask - 1
		}
	}

	return rank
}

// Select returns the index of the j-th set bit (0-indexed).
func (qb *QueryBitmap) Select(j int) (uint32, bool) {
	if j < 0 {
		return 0, false
	}

	remaining := j
	for maskIdx, mask := range qb.activeBlocks {
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit

			blockPop := int(qb.computeBlockPopcount(blockIdx))
			if blockPop > remaining {
				// The j-th bit is in this block
				start := blockIdx * BlockSize
				for w := start; w < start+BlockSize; w++ {
					word := qb.words[w]
					popcount := bits.OnesCount64(word)
					if popcount > remaining {
						return uint32(w)*WordBits + select64(word, remaining), true
					}
					remaining -= popcount
				}
			}
			remaining -= blockPop
			mask &= mask - 1
		}
	}
	return 0, false
}

// select64 returns the index of the j-th set bit within a word.
func select64(word uint64, j int) uint32 {
	for i := 0; i < j; i++ {
		word &= word - 1
	}
	return uint32(bits.TrailingZeros64(word))
}

// NextSet returns the next set bit from index i (inclusive).
// Uses activeBlocks bitset to skip 64 empty blocks at once.
func (qb *QueryBitmap) NextSet(i uint32) (uint32, bool) {
	if i >= qb.universeSize {
		return 0, false
	}

	wordIdx := int(i / WordBits)
	if wordIdx >= len(qb.words) {
		return 0, false
	}

	blockIdx := wordIdx / BlockSize

	// Check current block if active
	if qb.isBlockActive(blockIdx) {
		// Check partial first word
		word := qb.words[wordIdx] >> (i % WordBits)
		if word != 0 {
			return i + uint32(bits.TrailingZeros64(word)), true
		}

		// Check rest of current block
		blockEnd := (blockIdx + 1) * BlockSize
		for w := wordIdx + 1; w < blockEnd && w < len(qb.words); w++ {
			if qb.words[w] != 0 {
				return uint32(w*WordBits) + uint32(bits.TrailingZeros64(qb.words[w])), true
			}
		}
	}

	// Scan subsequent blocks using activeBlocks bitset
	blockIdx++
	maskIdx := blockIdx / BlocksPerMaskWord
	maskBit := blockIdx % BlocksPerMaskWord

	for maskIdx < len(qb.activeBlocks) {
		// Mask off blocks before our starting position
		mask := qb.activeBlocks[maskIdx] >> maskBit << maskBit

		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			nextBlock := maskIdx*BlocksPerMaskWord + bit

			// Scan words in this block
			start := nextBlock * BlockSize
			for w := start; w < start+BlockSize && w < len(qb.words); w++ {
				if qb.words[w] != 0 {
					return uint32(w*WordBits) + uint32(bits.TrailingZeros64(qb.words[w])), true
				}
			}
			mask &= mask - 1
		}

		maskIdx++
		maskBit = 0 // After first iteration, start from bit 0
	}

	return 0, false
}

// NextSetMany returns multiple set bits starting from index i.
// Uses activeBlocks bitset for efficient sparse iteration.
func (qb *QueryBitmap) NextSetMany(i uint32, buffer []uint32) (uint32, []uint32) {
	capacity := cap(buffer)
	if capacity == 0 {
		return 0, buffer[:0]
	}

	result := buffer[:capacity]
	size := 0

	wordIdx := int(i / WordBits)
	if wordIdx >= len(qb.words) {
		return 0, result[:0]
	}

	blockIdx := wordIdx / BlockSize

	// Process current block if active
	if qb.isBlockActive(blockIdx) {
		// First partial word
		word := qb.words[wordIdx] >> (i % WordBits)
		for word != 0 && size < capacity {
			result[size] = i + uint32(bits.TrailingZeros64(word))
			size++
			word &= word - 1
		}
		if size == capacity {
			return result[size-1], result[:size]
		}

		// Rest of current block
		blockEnd := (blockIdx + 1) * BlockSize
		for w := wordIdx + 1; w < blockEnd && w < len(qb.words); w++ {
			word = qb.words[w]
			baseID := uint32(w * WordBits)
			for word != 0 && size < capacity {
				result[size] = baseID + uint32(bits.TrailingZeros64(word))
				size++
				word &= word - 1
			}
			if size == capacity {
				return result[size-1], result[:size]
			}
		}
	}

	// Scan subsequent blocks using activeBlocks bitset
	blockIdx++
	maskIdx := blockIdx / BlocksPerMaskWord
	maskBit := blockIdx % BlocksPerMaskWord

	for maskIdx < len(qb.activeBlocks) {
		mask := qb.activeBlocks[maskIdx] >> maskBit << maskBit

		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			nextBlock := maskIdx*BlocksPerMaskWord + bit

			start := nextBlock * BlockSize
			for w := start; w < start+BlockSize && w < len(qb.words); w++ {
				word := qb.words[w]
				baseID := uint32(w * WordBits)
				for word != 0 && size < capacity {
					result[size] = baseID + uint32(bits.TrailingZeros64(word))
					size++
					word &= word - 1
				}
				if size == capacity {
					return result[size-1], result[:size]
				}
			}
			mask &= mask - 1
		}

		maskIdx++
		maskBit = 0
	}

	if size > 0 {
		return result[size-1], result[:size]
	}
	return 0, result[:0]
}

// ForEach iterates over all set bits, calling fn for each.
// Returns early if fn returns false. Zero allocations.
// Uses activeBlocks bitset to skip 64 empty blocks at once.
func (qb *QueryBitmap) ForEach(fn func(uint32) bool) {
	for maskIdx, mask := range qb.activeBlocks {
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit

			start := blockIdx * BlockSize
			baseID := uint32(start * WordBits)

			for w := start; w < start+BlockSize; w++ {
				word := qb.words[w]
				for word != 0 {
					bitPos := bits.TrailingZeros64(word)
					if !fn(baseID + uint32(bitPos)) {
						return
					}
					word &= word - 1
				}
				baseID += WordBits
			}

			mask &= mask - 1
		}
	}
}

// ToSlice returns all set bits as a sorted slice.
func (qb *QueryBitmap) ToSlice(scratch []uint32) []uint32 {
	card := qb.Cardinality()
	if cap(scratch) < card {
		scratch = make([]uint32, 0, card)
	} else {
		scratch = scratch[:0]
	}

	for maskIdx, mask := range qb.activeBlocks {
		for mask != 0 {
			bit := bits.TrailingZeros64(mask)
			blockIdx := maskIdx*BlocksPerMaskWord + bit

			start := blockIdx * BlockSize
			baseID := uint32(start * WordBits)

			for w := start; w < start+BlockSize; w++ {
				word := qb.words[w]
				for word != 0 {
					bitPos := bits.TrailingZeros64(word)
					scratch = append(scratch, baseID+uint32(bitPos))
					word &= word - 1
				}
				baseID += WordBits
			}

			mask &= mask - 1
		}
	}

	return scratch
}

// ToSliceInto copies all set bits into dst. Alias for ToSlice.
func (qb *QueryBitmap) ToSliceInto(dst []uint32) []uint32 {
	return qb.ToSlice(dst)
}

// ToArrayInto is an alias for ToSlice (segment.Bitmap interface compatibility).
func (qb *QueryBitmap) ToArrayInto(dst []uint32) []uint32 {
	return qb.ToSlice(dst)
}

// CopyFrom copies all bits from src into qb.
func (qb *QueryBitmap) CopyFrom(src *QueryBitmap) {
	n := min(len(qb.words), len(src.words))
	copy(qb.words[:n], src.words[:n])
	for i := n; i < len(qb.words); i++ {
		qb.words[i] = 0
	}

	nm := min(len(qb.activeBlocks), len(src.activeBlocks))
	copy(qb.activeBlocks[:nm], src.activeBlocks[:nm])
	for m := nm; m < len(qb.activeBlocks); m++ {
		qb.activeBlocks[m] = 0
	}

	nb := min(len(qb.blockPopcounts), len(src.blockPopcounts))
	copy(qb.blockPopcounts[:nb], src.blockPopcounts[:nb])
	for b := nb; b < len(qb.blockPopcounts); b++ {
		qb.blockPopcounts[b] = 0
	}

	qb.cardinality = src.cardinality
}

// Clone creates an independent copy of the bitmap.
func (qb *QueryBitmap) Clone() *QueryBitmap {
	cloned := &QueryBitmap{
		words:          make([]uint64, len(qb.words)),
		activeBlocks:   make([]uint64, len(qb.activeBlocks)),
		blockPopcounts: make([]uint16, len(qb.blockPopcounts)),
		universeSize:   qb.universeSize,
		numBlocks:      qb.numBlocks,
		cardinality:    qb.cardinality,
	}
	copy(cloned.words, qb.words)
	copy(cloned.activeBlocks, qb.activeBlocks)
	copy(cloned.blockPopcounts, qb.blockPopcounts)
	return cloned
}

// UniverseSize returns the maximum ID + 1.
func (qb *QueryBitmap) UniverseSize() uint32 {
	return qb.universeSize
}

// Words returns the underlying word slice.
func (qb *QueryBitmap) Words() []uint64 {
	return qb.words
}

// Density returns the fraction of bits set (0.0 to 1.0).
// Useful for density-adaptive execution strategies.
func (qb *QueryBitmap) Density() float64 {
	if qb.universeSize == 0 {
		return 0
	}
	return float64(qb.Cardinality()) / float64(qb.universeSize)
}

// ActiveBlockCount returns the number of non-empty blocks.
// Useful for estimating iteration cost.
func (qb *QueryBitmap) ActiveBlockCount() int {
	count := 0
	for _, mask := range qb.activeBlocks {
		count += bits.OnesCount64(mask)
	}
	return count
}

// CardinalityUint64 returns cardinality as uint64 (for segment.Bitmap interface).
func (qb *QueryBitmap) CardinalityUint64() uint64 {
	return uint64(qb.Cardinality())
}

// PopulateFromRoaring populates this QueryBitmap from a roaring bitmap.
// This is used to materialize a roaring bitmap into the SIMD-friendly format.
// Note: This allocates via rb.ToArray(). For zero-alloc, use PopulateFromRows.
func (qb *QueryBitmap) PopulateFromRoaring(rb interface{ ToArray() []uint32 }) {
	qb.Clear()
	qb.AddMany(rb.ToArray())
}

// PopulateFromRows populates this QueryBitmap from a sorted slice of row IDs.
// Zero allocations (uses existing backing storage).
// Optimized: batches block tracking updates.
func (qb *QueryBitmap) PopulateFromRows(rows []uint32) {
	qb.Clear()
	qb.AddMany(rows)
}

// ==============================================================================
// Pool
// ==============================================================================

// QueryBitmapPool is a pool of reusable QueryBitmaps. Thread-safe.
type QueryBitmapPool struct {
	pool         sync.Pool
	universeSize uint32
}

// NewQueryBitmapPool creates a new pool.
func NewQueryBitmapPool(universeSize uint32) *QueryBitmapPool {
	return &QueryBitmapPool{
		universeSize: universeSize,
		pool: sync.Pool{
			New: func() any {
				qb := New(universeSize)
				qb.ownedByPool = true
				return qb
			},
		},
	}
}

// Get retrieves a bitmap from the pool.
func (p *QueryBitmapPool) Get() *QueryBitmap {
	return p.pool.Get().(*QueryBitmap)
}

// Put returns a bitmap to the pool.
func (p *QueryBitmapPool) Put(qb *QueryBitmap) {
	if qb == nil || !qb.ownedByPool {
		return
	}
	qb.Clear()
	p.pool.Put(qb)
}
