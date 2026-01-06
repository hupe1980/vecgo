package engine

import "time"

// MetricsObserver defines the interface for observing engine events.
type MetricsObserver interface {
	// Write path
	OnInsert(latency time.Duration, err error)
	OnDelete(latency time.Duration, err error)
	OnWALWrite(latency time.Duration, bytes int)
	OnMemTableStatus(sizeBytes int64, percentFull float64)
	OnBackpressure(reason string)

	// Read path
	OnSearch(latency time.Duration, segmentType string, k int, retrieved int, err error)
	OnGet(latency time.Duration, err error)

	// Background operations
	OnFlush(duration time.Duration, rows int, bytes uint64, err error)
	OnCompaction(duration time.Duration, dropped int, created int, err error)
	OnBuild(duration time.Duration, indexType string, err error)
	OnStall(duration time.Duration, reason string)

	// Generic counts/gauges
	OnQueueDepth(name string, depth int)
	OnThroughput(name string, bytes int64)
}

// NoopMetricsObserver is a no-op implementation of MetricsObserver.
type NoopMetricsObserver struct{}

func (o *NoopMetricsObserver) OnInsert(time.Duration, error)                   {}
func (o *NoopMetricsObserver) OnDelete(time.Duration, error)                   {}
func (o *NoopMetricsObserver) OnWALWrite(time.Duration, int)                   {}
func (o *NoopMetricsObserver) OnMemTableStatus(int64, float64)                 {}
func (o *NoopMetricsObserver) OnBackpressure(string)                           {}
func (o *NoopMetricsObserver) OnSearch(time.Duration, string, int, int, error) {}
func (o *NoopMetricsObserver) OnGet(time.Duration, error)                      {}
func (o *NoopMetricsObserver) OnFlush(time.Duration, int, uint64, error)       {}
func (o *NoopMetricsObserver) OnCompaction(time.Duration, int, int, error)     {}
func (o *NoopMetricsObserver) OnBuild(time.Duration, string, error)            {}
func (o *NoopMetricsObserver) OnStall(time.Duration, string)                   {}
func (o *NoopMetricsObserver) OnQueueDepth(string, int)                        {}
func (o *NoopMetricsObserver) OnThroughput(string, int64)                      {}
