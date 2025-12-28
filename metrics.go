package vecgo

import (
	"sync/atomic"
	"time"
)

// MetricsCollector defines an interface for collecting operational metrics.
// Implement this interface to integrate with monitoring systems like Prometheus.
//
// Example Prometheus integration:
//
//	type PrometheusCollector struct {
//	    insertCounter   prometheus.Counter
//	    searchHistogram prometheus.Histogram
//	}
//
//	func (p *PrometheusCollector) RecordInsert(duration time.Duration, err error) {
//	    p.insertCounter.Inc()
//	    // ... record error state, duration, etc.
//	}
type MetricsCollector interface {
	// RecordInsert is called after each insert operation.
	// duration is the total time taken, err is nil if successful.
	RecordInsert(duration time.Duration, err error)

	// RecordBatchInsert is called after each batch insert operation.
	// count is the number of items attempted, failed is the number that failed,
	// duration is the total time taken.
	RecordBatchInsert(count, failed int, duration time.Duration)

	// RecordSearch is called after each search operation.
	// k is the number of neighbors requested, duration is the time taken,
	// err is nil if successful.
	RecordSearch(k int, duration time.Duration, err error)

	// RecordDelete is called after each delete operation.
	RecordDelete(duration time.Duration, err error)

	// RecordUpdate is called after each update operation.
	RecordUpdate(duration time.Duration, err error)
}

// NoopMetricsCollector is a no-op implementation of MetricsCollector.
// Use this when metrics collection is not needed.
type NoopMetricsCollector struct{}

func (NoopMetricsCollector) RecordInsert(time.Duration, error)         {}
func (NoopMetricsCollector) RecordBatchInsert(int, int, time.Duration) {}
func (NoopMetricsCollector) RecordSearch(int, time.Duration, error)    {}
func (NoopMetricsCollector) RecordDelete(time.Duration, error)         {}
func (NoopMetricsCollector) RecordUpdate(time.Duration, error)         {}

// BasicMetricsCollector provides simple in-memory metrics collection.
// Useful for debugging and basic monitoring without external dependencies.
type BasicMetricsCollector struct {
	InsertCount       atomic.Int64
	InsertErrors      atomic.Int64
	InsertTotalNanos  atomic.Int64
	BatchInsertCount  atomic.Int64
	BatchInsertItems  atomic.Int64
	BatchInsertFailed atomic.Int64
	SearchCount       atomic.Int64
	SearchErrors      atomic.Int64
	SearchTotalNanos  atomic.Int64
	DeleteCount       atomic.Int64
	DeleteErrors      atomic.Int64
	UpdateCount       atomic.Int64
	UpdateErrors      atomic.Int64
}

// RecordInsert implements MetricsCollector.
func (b *BasicMetricsCollector) RecordInsert(duration time.Duration, err error) {
	b.InsertCount.Add(1)
	b.InsertTotalNanos.Add(duration.Nanoseconds())
	if err != nil {
		b.InsertErrors.Add(1)
	}
}

// RecordBatchInsert implements MetricsCollector.
func (b *BasicMetricsCollector) RecordBatchInsert(count, failed int, duration time.Duration) {
	b.BatchInsertCount.Add(1)
	b.BatchInsertItems.Add(int64(count))
	b.BatchInsertFailed.Add(int64(failed))
}

// RecordSearch implements MetricsCollector.
func (b *BasicMetricsCollector) RecordSearch(k int, duration time.Duration, err error) {
	b.SearchCount.Add(1)
	b.SearchTotalNanos.Add(duration.Nanoseconds())
	if err != nil {
		b.SearchErrors.Add(1)
	}
}

// RecordDelete implements MetricsCollector.
func (b *BasicMetricsCollector) RecordDelete(duration time.Duration, err error) {
	b.DeleteCount.Add(1)
	if err != nil {
		b.DeleteErrors.Add(1)
	}
}

// RecordUpdate implements MetricsCollector.
func (b *BasicMetricsCollector) RecordUpdate(duration time.Duration, err error) {
	b.UpdateCount.Add(1)
	if err != nil {
		b.UpdateErrors.Add(1)
	}
}

// GetStats returns a snapshot of current metrics.
func (b *BasicMetricsCollector) GetStats() BasicMetricsStats {
	return BasicMetricsStats{
		InsertCount:       b.InsertCount.Load(),
		InsertErrors:      b.InsertErrors.Load(),
		InsertAvgNanos:    b.getAvgInsertNanos(),
		BatchInsertCount:  b.BatchInsertCount.Load(),
		BatchInsertItems:  b.BatchInsertItems.Load(),
		BatchInsertFailed: b.BatchInsertFailed.Load(),
		SearchCount:       b.SearchCount.Load(),
		SearchErrors:      b.SearchErrors.Load(),
		SearchAvgNanos:    b.getAvgSearchNanos(),
		DeleteCount:       b.DeleteCount.Load(),
		DeleteErrors:      b.DeleteErrors.Load(),
		UpdateCount:       b.UpdateCount.Load(),
		UpdateErrors:      b.UpdateErrors.Load(),
	}
}

func (b *BasicMetricsCollector) getAvgInsertNanos() int64 {
	count := b.InsertCount.Load()
	if count == 0 {
		return 0
	}
	return b.InsertTotalNanos.Load() / count
}

func (b *BasicMetricsCollector) getAvgSearchNanos() int64 {
	count := b.SearchCount.Load()
	if count == 0 {
		return 0
	}
	return b.SearchTotalNanos.Load() / count
}

// BasicMetricsStats is a snapshot of BasicMetricsCollector state.
type BasicMetricsStats struct {
	InsertCount       int64
	InsertErrors      int64
	InsertAvgNanos    int64
	BatchInsertCount  int64
	BatchInsertItems  int64
	BatchInsertFailed int64
	SearchCount       int64
	SearchErrors      int64
	SearchAvgNanos    int64
	DeleteCount       int64
	DeleteErrors      int64
	UpdateCount       int64
	UpdateErrors      int64
}
