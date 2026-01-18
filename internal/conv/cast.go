package conv

import (
	"fmt"
	"math"
)

// IntToUint32 converts int to uint32 safely.
func IntToUint32(v int) (uint32, error) {
	if v < 0 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint32 (negative)", v)
	}
	// On 64-bit systems, int can exceed uint32 max; on 32-bit, this is always false
	if uint64(v) > math.MaxUint32 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint32 (too large)", v)
	}
	return uint32(v), nil
}

// IntToUint64 converts int to uint64 safely.
func IntToUint64(v int) (uint64, error) {
	if v < 0 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint64 (negative)", v)
	}
	return uint64(v), nil
}

// Uint64ToInt converts uint64 to int safely.
func Uint64ToInt(v uint64) (int, error) {
	if v > uint64(math.MaxInt) {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int (too large)", v)
	}
	return int(v), nil
}

// Uint64ToUint32 converts uint64 to uint32 safely.
func Uint64ToUint32(v uint64) (uint32, error) {
	if v > math.MaxUint32 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint32 (too large)", v)
	}
	return uint32(v), nil
}

// Uint32ToInt converts uint32 to int safely.
func Uint32ToInt(v uint32) (int, error) {
	if uint64(v) > uint64(math.MaxInt) {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int (too large)", v)
	}
	return int(v), nil
}
