package manifest

import (
	"math"
	"sort"

	"github.com/hupe1980/vecgo/metadata"
)

// StatsCollector accumulates statistics during segment iteration.
// It is designed for single-threaded use during flush/compaction.
// Thread-safety: NOT thread-safe. Use one collector per segment.
//
// Enhanced features:
// - Log-scaled histograms with per-bin min/max
// - Categorical entropy computation
// - Vector distance-to-centroid statistics (radius95, cluster tightness)
// - Segment shape detection (sorted, append-only, clustered)
// - Filter entropy score
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
	sumVec    []float64   // For centroid computation (only if dim <= 256)
	vectors   [][]float32 // Stored vectors for distance computation (capped)
	dim       int
	trackVec  bool
	firstNorm bool

	// Field existence
	hasFields map[string]bool

	// Row counts
	totalRows   uint32
	liveRows    uint32
	deletedRows uint32

	// Shape detection
	timestampField string
	lastTimestamp  float64
	isSorted       bool
	isAppendOnly   bool
	hasDeletes     bool
	firstRow       bool
}

type numericAccum struct {
	min    float64
	max    float64
	sum    float64
	sumSq  float64
	count  uint32
	hasNaN bool
	first  bool
	values []float64 // For histogram computation (capped)
}

// MaxHistogramSamples limits memory for histogram building.
// We sample values and compute histogram at finalize time.
const MaxHistogramSamples = 100000

// MaxVectorSamples limits memory for distance-to-centroid computation.
const MaxVectorSamples = 10000

type categoricalAccum struct {
	counts map[string]uint32
}

// MaxCategoricalDistinct limits memory for categorical tracking.
// Fields with more distinct values switch to count-only mode.
const MaxCategoricalDistinct = 10000

// TopKLimit is the number of top values to keep.
const TopKLimit = 16

// NewStatsCollector creates a new stats collector.
// If dim <= 256 and trackVector is true, it will compute centroid and radius stats.
func NewStatsCollector(dim int, trackVector bool) *StatsCollector {
	sc := &StatsCollector{
		numeric:      make(map[string]*numericAccum),
		categorical:  make(map[string]*categoricalAccum),
		hasFields:    make(map[string]bool),
		dim:          dim,
		trackVec:     trackVector && dim <= 256,
		minNorm:      math.MaxFloat32,
		maxNorm:      0,
		firstNorm:    true,
		isSorted:     true, // Assume sorted until proven otherwise
		isAppendOnly: true, // Assume append-only until proven otherwise
		firstRow:     true,
	}

	if sc.trackVec {
		sc.sumVec = make([]float64, dim)
		sc.vectors = make([][]float32, 0, MaxVectorSamples)
	}

	return sc
}

// Add processes a single row during iteration.
func (sc *StatsCollector) Add(vec []float32, md metadata.Document) {
	sc.totalRows++
	sc.liveRows++

	// Process metadata fields
	for key, val := range md {
		sc.hasFields[key] = true

		switch val.Kind {
		case metadata.KindInt:
			v := float64(val.I64)
			sc.addNumeric(key, v)
			sc.checkTimestampSorting(key, v)
		case metadata.KindFloat:
			v := val.F64
			if math.IsNaN(v) {
				sc.addNumericNaN(key)
			} else {
				sc.addNumeric(key, v)
				sc.checkTimestampSorting(key, v)
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

	sc.firstRow = false

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

		// Store vector for distance computation (capped)
		if len(sc.vectors) < MaxVectorSamples {
			vecCopy := make([]float32, len(vec))
			copy(vecCopy, vec)
			sc.vectors = append(sc.vectors, vecCopy)
		}
	}
}

// AddDeleted processes a deleted row (tombstone).
func (sc *StatsCollector) AddDeleted() {
	sc.totalRows++
	sc.deletedRows++
	sc.hasDeletes = true
	sc.isAppendOnly = false
}

// checkTimestampSorting detects if data is sorted by a timestamp-like field.
func (sc *StatsCollector) checkTimestampSorting(key string, val float64) {
	// Heuristic: look for fields that might be timestamps
	isTimestampField := key == "timestamp" || key == "ts" || key == "created_at" ||
		key == "updated_at" || key == "time" || key == "date"

	if !isTimestampField {
		return
	}

	if sc.firstRow {
		sc.timestampField = key
		sc.lastTimestamp = val
		return
	}

	// Only track one timestamp field
	if sc.timestampField != key {
		return
	}

	// Check if still sorted
	if val < sc.lastTimestamp {
		sc.isSorted = false
	}
	sc.lastTimestamp = val
}

// l2Norm computes the L2 norm of a vector.
func l2Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// l2Distance computes the L2 distance between two vectors.
func l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
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

	// Track sum and sum of squares for variance
	acc.sum += val
	acc.sumSq += val * val
	acc.count++

	// Track values for histogram (with cap)
	if len(acc.values) < MaxHistogramSamples {
		acc.values = append(acc.values, val)
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
	if sc.totalRows == 0 {
		return nil
	}

	stats := &SegmentStats{
		TotalRows:   sc.totalRows,
		LiveRows:    sc.liveRows,
		Numeric:     make(map[string]NumericFieldStats, len(sc.numeric)),
		Categorical: make(map[string]CategoricalStats, len(sc.categorical)),
		HasFields:   sc.hasFields,
	}

	// Compute deleted ratio
	if sc.totalRows > 0 {
		stats.DeletedRatio = float32(sc.deletedRows) / float32(sc.totalRows)
	}

	// Numeric fields with log-scaled histograms
	var numericEntropySum float64
	var numericCount int
	for key, acc := range sc.numeric {
		nfs := NumericFieldStats{
			Min:    acc.min,
			Max:    acc.max,
			HasNaN: acc.hasNaN,
			Sum:    acc.sum,
			SumSq:  acc.sumSq,
			Count:  acc.count,
		}

		// Build log-scaled histogram from sampled values
		if len(acc.values) > 0 && acc.max > acc.min {
			sc.buildLogScaledHistogram(&nfs, acc.values)

			// Compute numeric field entropy from histogram
			entropy := sc.computeHistogramEntropy(&nfs)
			numericEntropySum += entropy
			numericCount++
		}

		stats.Numeric[key] = nfs
	}

	// Categorical fields with entropy
	var catEntropySum float64
	var catCount int
	for key, acc := range sc.categorical {
		cs := CategoricalStats{
			DistinctCount: uint32(len(acc.counts)),
		}

		// Find dominant value and compute entropy
		var maxVal string
		var maxCount uint32
		var totalCount uint32
		for val, count := range acc.counts {
			totalCount += count
			if count > maxCount {
				maxCount = count
				maxVal = val
			}
		}

		// Compute purity ratio
		if totalCount > 0 {
			cs.DominantValue = maxVal
			cs.DominantRatio = float32(maxCount) / float32(totalCount)

			// Compute Shannon entropy (normalized to [0,1])
			cs.Entropy = sc.computeCategoricalEntropy(acc.counts, totalCount)
			catEntropySum += float64(cs.Entropy)
			catCount++
		}

		// Extract top-k
		if len(acc.counts) <= TopKLimit {
			cs.TopK = make([]ValueFreq, 0, len(acc.counts))
			for val, count := range acc.counts {
				cs.TopK = append(cs.TopK, ValueFreq{Value: val, Count: count})
			}
		} else {
			cs.TopK = topKByCount(acc.counts, TopKLimit)
		}

		// Build Bloom filter for high-cardinality fields
		// Skip if TopK already covers all values (low cardinality)
		if len(acc.counts) > TopKLimit {
			// Create Bloom filter with 1% false positive rate
			// Memory: ~10 bits per value ≈ 1.25 bytes per value
			cs.Bloom = NewBloomFilterForSize(len(acc.counts))
			for val := range acc.counts {
				cs.Bloom.Add(val)
			}
		}

		stats.Categorical[key] = cs
	}

	// Compute overall filter entropy (weighted average)
	totalFields := numericCount + catCount
	if totalFields > 0 {
		// Weight categorical more heavily (typically more selective)
		catWeight := 0.6
		numWeight := 0.4
		if numericCount == 0 {
			catWeight = 1.0
			numWeight = 0.0
		} else if catCount == 0 {
			catWeight = 0.0
			numWeight = 1.0
		}
		avgNumericEntropy := 0.0
		if numericCount > 0 {
			avgNumericEntropy = numericEntropySum / float64(numericCount)
		}
		avgCatEntropy := 0.0
		if catCount > 0 {
			avgCatEntropy = catEntropySum / float64(catCount)
		}
		stats.FilterEntropy = float32(avgNumericEntropy*numWeight + avgCatEntropy*catWeight)
	}

	// Vector stats with distance-to-centroid
	if sc.trackVec && sc.vecCount > 0 {
		stats.Vector = sc.finalizeVectorStats()
	}

	// Shape stats
	stats.Shape = &ShapeStats{
		IsSortedByTimestamp: sc.isSorted && sc.timestampField != "",
		TimestampField:      sc.timestampField,
		IsAppendOnly:        sc.isAppendOnly && !sc.hasDeletes,
	}

	// Check if vectors are clustered
	if stats.Vector != nil && stats.Vector.AvgDistanceToCentroid > 0 {
		// Clustered if avg distance is small relative to norm
		if stats.Vector.MeanNorm > 0 {
			relativeSpread := stats.Vector.AvgDistanceToCentroid / stats.Vector.MeanNorm
			stats.Shape.IsClustered = relativeSpread < 0.5

			// Cluster tightness: 1 - (spread / avgDist)
			// Higher = tighter cluster
			if stats.Vector.Radius95 > 0 {
				spread := stats.Vector.Radius95 - stats.Vector.AvgDistanceToCentroid
				if spread < 0 {
					spread = 0
				}
				stats.Shape.ClusterTightness = 1.0 - float32(math.Min(1.0, float64(spread/stats.Vector.AvgDistanceToCentroid)))
			}
		}
	}

	return stats
}

// buildLogScaledHistogram builds a log-scaled histogram with per-bin min/max.
func (sc *StatsCollector) buildLogScaledHistogram(nfs *NumericFieldStats, values []float64) {
	segmentRange := nfs.Max - nfs.Min
	if segmentRange <= 0 {
		return
	}

	// Initialize per-bin tracking
	binMinInit := make([]bool, HistogramBins)
	for i := range nfs.HistogramMin {
		nfs.HistogramMin[i] = math.MaxFloat64
		nfs.HistogramMax[i] = -math.MaxFloat64
	}

	// Compute log-scaled bin boundaries
	binBoundaries := make([]float64, HistogramBins+1)
	binBoundaries[0] = nfs.Min
	for i := 1; i <= HistogramBins; i++ {
		t := math.Log(1+float64(i)) / math.Log(17)
		binBoundaries[i] = nfs.Min + segmentRange*t
	}

	// Assign values to bins
	for _, v := range values {
		// Binary search for bin (log-scaled boundaries)
		bin := sort.Search(HistogramBins, func(i int) bool {
			return binBoundaries[i+1] > v
		})
		if bin >= HistogramBins {
			bin = HistogramBins - 1
		}
		if bin < 0 {
			bin = 0
		}

		nfs.Histogram[bin]++

		// Track per-bin min/max
		if !binMinInit[bin] || v < nfs.HistogramMin[bin] {
			nfs.HistogramMin[bin] = v
			binMinInit[bin] = true
		}
		if v > nfs.HistogramMax[bin] {
			nfs.HistogramMax[bin] = v
		}
	}

	// Clean up uninitialized bins
	for i := range nfs.HistogramMin {
		if nfs.Histogram[i] == 0 {
			nfs.HistogramMin[i] = 0
			nfs.HistogramMax[i] = 0
		}
	}
}

// computeHistogramEntropy computes entropy from histogram (normalized to [0,1]).
func (sc *StatsCollector) computeHistogramEntropy(nfs *NumericFieldStats) float64 {
	var total uint32
	for _, c := range nfs.Histogram {
		total += c
	}
	if total == 0 {
		return 0
	}

	var entropy float64
	for _, count := range nfs.Histogram {
		if count == 0 {
			continue
		}
		p := float64(count) / float64(total)
		entropy -= p * math.Log2(p)
	}

	// Normalize by max entropy (uniform distribution)
	maxEntropy := math.Log2(float64(HistogramBins))
	if maxEntropy > 0 {
		entropy /= maxEntropy
	}

	return entropy
}

// computeCategoricalEntropy computes Shannon entropy (normalized to [0,1]).
func (sc *StatsCollector) computeCategoricalEntropy(counts map[string]uint32, total uint32) float32 {
	if total == 0 || len(counts) <= 1 {
		return 0
	}

	var entropy float64
	for _, count := range counts {
		if count == 0 {
			continue
		}
		p := float64(count) / float64(total)
		entropy -= p * math.Log2(p)
	}

	// Normalize by max entropy (uniform distribution over distinct values)
	maxEntropy := math.Log2(float64(len(counts)))
	if maxEntropy > 0 {
		entropy /= maxEntropy
	}

	return float32(entropy)
}

// finalizeVectorStats computes final vector statistics including distance-to-centroid.
func (sc *StatsCollector) finalizeVectorStats() *VectorStats {
	meanNorm := float32(sc.normSum / float64(sc.vecCount))

	vs := &VectorStats{
		MinNorm:  sc.minNorm,
		MaxNorm:  sc.maxNorm,
		MeanNorm: meanNorm,
	}

	// Compute centroid (quantized to int8)
	centroid := make([]float32, sc.dim) // Full precision for distance computation
	centroidInt8 := make([]int8, sc.dim)
	for i := 0; i < sc.dim; i++ {
		mean := sc.sumVec[i] / float64(sc.vecCount)
		centroid[i] = float32(mean)

		// Quantize to int8 range [-127, 127]
		q := int(mean * 127)
		if q > 127 {
			q = 127
		}
		if q < -127 {
			q = -127
		}
		centroidInt8[i] = int8(q)
	}
	vs.Centroid = centroidInt8

	// Compute distance-to-centroid statistics from sampled vectors
	if len(sc.vectors) > 0 {
		distances := make([]float32, len(sc.vectors))
		var distSum float64
		var maxDist float32

		for i, vec := range sc.vectors {
			dist := l2Distance(vec, centroid)
			distances[i] = dist
			distSum += float64(dist)
			if dist > maxDist {
				maxDist = dist
			}
		}

		vs.AvgDistanceToCentroid = float32(distSum / float64(len(distances)))
		vs.RadiusMax = maxDist

		// Compute 95th percentile (Radius95)
		sort.Slice(distances, func(i, j int) bool {
			return distances[i] < distances[j]
		})
		p95Index := int(float64(len(distances)) * 0.95)
		if p95Index >= len(distances) {
			p95Index = len(distances) - 1
		}
		vs.Radius95 = distances[p95Index]
	}

	return vs
}

// topKByCount extracts the top k values by frequency.
func topKByCount(counts map[string]uint32, k int) []ValueFreq {
	result := make([]ValueFreq, 0, len(counts))
	for val, count := range counts {
		result = append(result, ValueFreq{Value: val, Count: count})
	}

	// Partial sort: only need top k
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
