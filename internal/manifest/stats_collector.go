package manifest

import (
	"math"

	"github.com/hupe1980/vecgo/metadata"
)

// StatsCollector accumulates statistics during segment iteration.
// It is designed for single-threaded use during flush/compaction.
// Thread-safety: NOT thread-safe. Use one collector per segment.
type StatsCollector struct {
	// Numeric field tracking
	numeric map[string]*numericAccum

	// Categorical field tracking
	categorical map[string]*categoricalAccum

	// Vector tracking
	vecCount  uint64
	normSum   float64
	minNorm   float32
	maxNorm   float32
	sumVec    []float64 // For centroid computation (only if dim <= 256)
	dim       int
	trackVec  bool
	firstNorm bool

	// Field existence
	hasFields map[string]bool

	// Row count
	rowCount uint32
}

type numericAccum struct {
	min    float64
	max    float64
	hasNaN bool
	first  bool
}

type categoricalAccum struct {
	counts map[string]uint32
}

// MaxCategoricalDistinct limits memory for categorical tracking.
// Fields with more distinct values switch to count-only mode.
const MaxCategoricalDistinct = 10000

// TopKLimit is the number of top values to keep.
const TopKLimit = 16

// NewStatsCollector creates a new stats collector.
// If dim <= 256 and trackVector is true, it will compute centroid.
func NewStatsCollector(dim int, trackVector bool) *StatsCollector {
	sc := &StatsCollector{
		numeric:     make(map[string]*numericAccum),
		categorical: make(map[string]*categoricalAccum),
		hasFields:   make(map[string]bool),
		dim:         dim,
		trackVec:    trackVector && dim <= 256,
		minNorm:     math.MaxFloat32,
		maxNorm:     0,
		firstNorm:   true,
	}

	if sc.trackVec {
		sc.sumVec = make([]float64, dim)
	}

	return sc
}

// Add processes a single row during iteration.
func (sc *StatsCollector) Add(vec []float32, md metadata.Document) {
	sc.rowCount++

	// Process metadata fields
	for key, val := range md {
		sc.hasFields[key] = true

		switch val.Kind {
		case metadata.KindInt:
			sc.addNumeric(key, float64(val.I64))
		case metadata.KindFloat:
			v := val.F64
			if math.IsNaN(v) {
				sc.addNumericNaN(key)
			} else {
				sc.addNumeric(key, v)
			}
		case metadata.KindString:
			sc.addCategorical(key, val.StringValue())
		case metadata.KindBool:
			// Track as categorical "true"/"false"
			if val.B {
				sc.addCategorical(key, "true")
			} else {
				sc.addCategorical(key, "false")
			}
		}
	}

	// Process vector
	if sc.trackVec && len(vec) == sc.dim {
		norm := l2Norm(vec)

		if sc.firstNorm {
			sc.minNorm = norm
			sc.maxNorm = norm
			sc.firstNorm = false
		} else {
			if norm < sc.minNorm {
				sc.minNorm = norm
			}
			if norm > sc.maxNorm {
				sc.maxNorm = norm
			}
		}

		sc.normSum += float64(norm)
		sc.vecCount++

		// Accumulate for centroid
		for i, v := range vec {
			sc.sumVec[i] += float64(v)
		}
	}
}

// l2Norm computes the L2 norm of a vector.
func l2Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

func (sc *StatsCollector) addNumeric(key string, val float64) {
	acc, ok := sc.numeric[key]
	if !ok {
		acc = &numericAccum{min: val, max: val, first: true}
		sc.numeric[key] = acc
	}

	if acc.first {
		acc.min = val
		acc.max = val
		acc.first = false
	} else {
		if val < acc.min {
			acc.min = val
		}
		if val > acc.max {
			acc.max = val
		}
	}
}

func (sc *StatsCollector) addNumericNaN(key string) {
	acc, ok := sc.numeric[key]
	if !ok {
		acc = &numericAccum{first: true}
		sc.numeric[key] = acc
	}
	acc.hasNaN = true
}

func (sc *StatsCollector) addCategorical(key, val string) {
	acc, ok := sc.categorical[key]
	if !ok {
		acc = &categoricalAccum{counts: make(map[string]uint32)}
		sc.categorical[key] = acc
	}

	// Memory limit: if too many distinct values, stop tracking individual values
	if len(acc.counts) < MaxCategoricalDistinct {
		acc.counts[val]++
	}
}

// Finalize computes the final SegmentStats.
func (sc *StatsCollector) Finalize() *SegmentStats {
	if sc.rowCount == 0 {
		return nil
	}

	stats := &SegmentStats{
		Numeric:     make(map[string]NumericFieldStats, len(sc.numeric)),
		Categorical: make(map[string]CategoricalStats, len(sc.categorical)),
		HasFields:   sc.hasFields,
	}

	// Numeric fields
	for key, acc := range sc.numeric {
		stats.Numeric[key] = NumericFieldStats{
			Min:    acc.min,
			Max:    acc.max,
			HasNaN: acc.hasNaN,
		}
	}

	// Categorical fields
	for key, acc := range sc.categorical {
		cs := CategoricalStats{
			DistinctCount: uint32(len(acc.counts)),
		}

		// Extract top-k
		if len(acc.counts) <= TopKLimit {
			// Small enough: include all
			cs.TopK = make([]ValueFreq, 0, len(acc.counts))
			for val, count := range acc.counts {
				cs.TopK = append(cs.TopK, ValueFreq{Value: val, Count: count})
			}
		} else {
			// Find top-k by count
			cs.TopK = topKByCount(acc.counts, TopKLimit)
		}

		stats.Categorical[key] = cs
	}

	// Vector stats
	if sc.trackVec && sc.vecCount > 0 {
		meanNorm := float32(sc.normSum / float64(sc.vecCount))

		vs := &VectorStats{
			MinNorm:  sc.minNorm,
			MaxNorm:  sc.maxNorm,
			MeanNorm: meanNorm,
		}

		// Compute centroid (quantized to int8)
		if sc.dim <= 256 {
			centroid := make([]int8, sc.dim)
			for i := 0; i < sc.dim; i++ {
				// Mean value
				mean := sc.sumVec[i] / float64(sc.vecCount)
				// Quantize to int8 range [-127, 127]
				// Assumes values are roughly in [-1, 1] range (typical for normalized embeddings)
				q := int(mean * 127)
				if q > 127 {
					q = 127
				}
				if q < -127 {
					q = -127
				}
				centroid[i] = int8(q)
			}
			vs.Centroid = centroid
		}

		stats.Vector = vs
	}

	return stats
}

// topKByCount extracts the top k values by frequency.
func topKByCount(counts map[string]uint32, k int) []ValueFreq {
	// Simple approach: sort all, take top k
	// For production with very large maps, use heap
	result := make([]ValueFreq, 0, len(counts))
	for val, count := range counts {
		result = append(result, ValueFreq{Value: val, Count: count})
	}

	// Partial sort: only need top k
	// Simple selection sort for small k
	for i := 0; i < k && i < len(result); i++ {
		maxIdx := i
		for j := i + 1; j < len(result); j++ {
			if result[j].Count > result[maxIdx].Count {
				maxIdx = j
			}
		}
		if maxIdx != i {
			result[i], result[maxIdx] = result[maxIdx], result[i]
		}
	}

	if len(result) > k {
		result = result[:k]
	}
	return result
}
