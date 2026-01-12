// Package conv provides safe type conversion utilities.
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
	if v > math.MaxUint32 {
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

// IntToUint16 converts int to uint16 safely.
func IntToUint16(v int) (uint16, error) {
	if v < 0 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint16 (negative)", v)
	}
	if v > math.MaxUint16 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint16 (too large)", v)
	}
	return uint16(v), nil
}

// Int64ToUint64 converts int64 to uint64 safely.
func Int64ToUint64(v int64) (uint64, error) {
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

// Uint64ToInt64 converts uint64 to int64 safely.
func Uint64ToInt64(v uint64) (int64, error) {
	if v > math.MaxInt64 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int64 (too large)", v)
	}
	return int64(v), nil
}

// Uint64ToUint32 converts uint64 to uint32 safely.
func Uint64ToUint32(v uint64) (uint32, error) {
	if v > math.MaxUint32 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to uint32 (too large)", v)
	}
	return uint32(v), nil
}

// Uint32ToInt32 converts uint32 to int32 safely.
func Uint32ToInt32(v uint32) (int32, error) {
	if v > math.MaxInt32 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int32 (too large)", v)
	}
	return int32(v), nil
}

// IntToInt32 converts int to int32 safely.
func IntToInt32(v int) (int32, error) {
	if v < math.MinInt32 || v > math.MaxInt32 {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int32", v)
	}
	return int32(v), nil
}

// Uint32ToInt converts uint32 to int safely.
func Uint32ToInt(v uint32) (int, error) {
	if uint64(v) > uint64(math.MaxInt) {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int (too large)", v)
	}
	return int(v), nil
}

// Int64ToInt converts int64 to int safely.
func Int64ToInt(v int64) (int, error) {
	if v > int64(math.MaxInt) || v < int64(math.MinInt) {
		return 0, fmt.Errorf("integer overflow: %d cannot be converted to int", v)
	}
	return int(v), nil
}
