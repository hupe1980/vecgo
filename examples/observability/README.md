# Observability Example

This example demonstrates how to use the `MetricsObserver` interface to gain visibility into the Vecgo engine.

## Running the Example

```bash
go run main.go
```

The example uses a `SimpleObserver` which simply logs metrics to `stdout`.

## Prometheus Integration

To integrate with Prometheus, implement `MetricsObserver` using Prometheus counters and histograms.

Example snippet (requires `github.com/prometheus/client_golang`):

```go
type PrometheusObserver struct {
    engine.NoopMetricsObserver
    
    insertDuration *prometheus.HistogramVec
    insertCount    *prometheus.CounterVec
    // ... define other metrics
}

func (p *PrometheusObserver) OnInsert(duration time.Duration, err error) {
    status := "success"
    if err != nil {
        status = "error"
    }
    p.insertDuration.WithLabelValues(status).Observe(duration.Seconds())
    p.insertCount.WithLabelValues(status).Inc()
}

// ... implement other methods
```

Inject this observer into the engine options:

```go
opts := engine.Options{
    Metrics: &PrometheusObserver{},
}
```
