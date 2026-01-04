package engine

import "time"

// MetricsObserver defines the interface for observing engine events.
type MetricsObserver interface {
	// OnFlush is called when a flush completes.
	OnFlush(duration time.Duration, rows int, err error)

	// OnCompaction is called when a compaction completes.
	OnCompaction(duration time.Duration, inputSegments int, outputRows int, err error)

	// OnBuild is called when an index build completes.
	OnBuild(duration time.Duration, indexType string, err error)

	// OnQueueDepth reports the depth of a background queue.
	OnQueueDepth(name string, depth int)

	// OnThroughput reports bytes processed.
	OnThroughput(name string, bytes int64)
}

// NoopMetricsObserver is a no-op implementation of MetricsObserver.
type NoopMetricsObserver struct{}

func (o *NoopMetricsObserver) OnFlush(duration time.Duration, rows int, err error) {}
func (o *NoopMetricsObserver) OnCompaction(duration time.Duration, inputSegments int, outputRows int, err error) {
}
func (o *NoopMetricsObserver) OnBuild(duration time.Duration, indexType string, err error) {}
func (o *NoopMetricsObserver) OnQueueDepth(name string, depth int)                         {}
func (o *NoopMetricsObserver) OnThroughput(name string, bytes int64)                       {}
