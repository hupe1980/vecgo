package metadata

import (
	"encoding/binary"
	"errors"
)

// MatchesBinary checks if the provided binary encoded metadata matches all filters in the set.
// This avoids full unmarshalling of the metadata document.
func (fs *FilterSet) MatchesBinary(data []byte) (bool, error) {
	if len(data) == 0 {
		return false, errors.New("empty data")
	}

	count, n := binary.Uvarint(data)
	if n <= 0 {
		return false, errors.New("invalid metadata length")
	}
	data = data[n:]

	// Track which filters have been matched.
	// Since all filters in FilterSet are ANDed, and must exist, we need to find N filters.
	// For small N (typical case), a simple bitmask or counter is fine.
	// But FilterSet stores filters as slice.

	// Use cached filterMap for O(1) lookup (avoids allocation on each call).
	// Optimization: If FilterSet has 1 filter, avoid map.
	var singleFilter *Filter
	var filterMap map[string]*Filter

	if len(fs.Filters) == 1 {
		singleFilter = &fs.Filters[0]
	} else {
		// Lazy-initialize cached filterMap
		if fs.filterMap == nil {
			fs.filterMap = make(map[string]*Filter, len(fs.Filters))
			for i := range fs.Filters {
				fs.filterMap[fs.Filters[i].Key] = &fs.Filters[i]
			}
		}
		filterMap = fs.filterMap
	}

	matchedCount := 0
	targetCount := len(fs.Filters)

	for i := uint64(0); i < count; i++ {
		// Stop early if we matched everything
		if matchedCount == targetCount {
			return true, nil
		}

		// Read Key
		kLen, n := binary.Uvarint(data)
		if n <= 0 {
			return false, errors.New("invalid key length")
		}
		data = data[n:]
		if uint64(len(data)) < kLen {
			return false, errors.New("short buffer for key")
		}

		// Check key presence without allocating string if possible?
		// We still need to compare bytes.
		keyBytes := data[:kLen]
		data = data[kLen:]

		// Check against filters
		var f *Filter
		if singleFilter != nil {
			if string(keyBytes) == singleFilter.Key {
				f = singleFilter
			}
		} else {
			f = filterMap[string(keyBytes)]
		}

		if f != nil {
			// Key matches a filter.
			// Decode Value and check match.
			val, bytesRead, err := parseValueN(data)
			if err != nil {
				return false, err
			}
			data = data[bytesRead:]

			if !f.MatchesValue(val) {
				return false, nil // Short circuit: AND condition failed
			}
			matchedCount++
		} else {
			// Skip Value
			remaining, err := skipValue(data)
			if err != nil {
				return false, err
			}
			data = remaining
		}
	}

	// Did we find all required keys?
	return matchedCount == targetCount, nil
}

func skipValue(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, errors.New("short buffer for value kind")
	}
	kind := Kind(data[0])
	data = data[1:]

	switch kind {
	case KindNull:
		// No payload
	case KindInt:
		_, n := binary.Varint(data)
		if n <= 0 {
			return nil, errors.New("invalid int value")
		}
		data = data[n:]
	case KindFloat:
		if len(data) < 8 {
			return nil, errors.New("short buffer for float")
		}
		data = data[8:]
	case KindString:
		sLen, n := binary.Uvarint(data)
		if n <= 0 {
			return nil, errors.New("invalid string length")
		}
		data = data[n:]
		if uint64(len(data)) < sLen {
			return nil, errors.New("short buffer for string")
		}
		data = data[sLen:]
	case KindBool:
		if len(data) == 0 {
			return nil, errors.New("short buffer for bool")
		}
		data = data[1:]
	case KindArray:
		aLen, n := binary.Uvarint(data)
		if n <= 0 {
			return nil, errors.New("invalid array length")
		}
		data = data[n:]
		for i := uint64(0); i < aLen; i++ {
			remaining, err := skipValue(data)
			if err != nil {
				return nil, err
			}
			data = remaining
		}
	default:
		return nil, errors.New("unknown metadata kind")
	}
	return data, nil
}
