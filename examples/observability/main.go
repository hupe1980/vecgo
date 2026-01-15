package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/testutil"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const dim = 128

// Use deterministic RNG for reproducible examples
var rng = testutil.NewRNG(42)

// PrometheusObserver implements vecgo.MetricsObserver
type PrometheusObserver struct {
	opLatency     *prometheus.HistogramVec
	memTableBytes prometheus.Gauge
	memTablePct   prometheus.Gauge
	backpressure  prometheus.Counter
	queueDepth    *prometheus.GaugeVec
	compactions   *prometheus.CounterVec
	flushes       prometheus.Counter
	writes        *prometheus.CounterVec // To cross-check QPS
}

func NewPrometheusObserver() *PrometheusObserver {
	o := &PrometheusObserver{
		opLatency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vecgo_operation_latency_seconds",
			Help:    "Latency of engine operations",
			Buckets: prometheus.DefBuckets, // Use default buckets for simplicity
		}, []string{"op", "status"}),
		memTableBytes: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "vecgo_memtable_size_bytes",
			Help: "Current size of memtable in bytes",
		}),
		memTablePct: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "vecgo_memtable_usage_ratio",
			Help: "Current usage of memtable as a ratio (0.0-1.0)",
		}),
		backpressure: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vecgo_backpressure_events_total",
			Help: "Total count of backpressure events",
		}),
		queueDepth: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "vecgo_queue_depth",
			Help: "Depth of various queues",
		}, []string{"queue"}),
		compactions: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vecgo_compactions_total",
			Help: "Total compactions completed",
		}, []string{"status"}),
		flushes: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "vecgo_flushes_total",
			Help: "Total flushes completed",
		}),
		writes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vecgo_writes_total",
			Help: "Total writes processed",
		}, []string{"type"}),
	}

	prometheus.MustRegister(o.opLatency)
	prometheus.MustRegister(o.memTableBytes)
	prometheus.MustRegister(o.memTablePct)
	prometheus.MustRegister(o.backpressure)
	prometheus.MustRegister(o.queueDepth)
	prometheus.MustRegister(o.compactions)
	prometheus.MustRegister(o.flushes)
	prometheus.MustRegister(o.writes)
	return o
}

func (o *PrometheusObserver) OnInsert(d time.Duration, err error) {
	status := "success"
	if err != nil {
		status = "error"
	}
	o.opLatency.WithLabelValues("insert", status).Observe(d.Seconds())
	o.writes.WithLabelValues("insert").Inc()
}

func (o *PrometheusObserver) OnDelete(d time.Duration, err error) {
	status := "success"
	if err != nil {
		status = "error"
	}
	o.opLatency.WithLabelValues("delete", status).Observe(d.Seconds())
	o.writes.WithLabelValues("delete").Inc()
}

func (o *PrometheusObserver) OnSearch(d time.Duration, segType string, k int, n int, err error) {
	status := "success"
	if err != nil {
		status = "error"
	}
	// segType might be useful, but let's keep cardinality low for opLatency
	o.opLatency.WithLabelValues("search", status).Observe(d.Seconds())
}

func (o *PrometheusObserver) OnMemTableStatus(bytes int64, pct float64) {
	o.memTableBytes.Set(float64(bytes))
	o.memTablePct.Set(pct)
}

func (o *PrometheusObserver) OnBackpressure(reason string) {
	o.backpressure.Inc()
}

func (o *PrometheusObserver) OnQueueDepth(name string, depth int) {
	o.queueDepth.WithLabelValues(name).Set(float64(depth))
}

func (o *PrometheusObserver) OnCompaction(d time.Duration, dropped, created int, err error) {
	status := "success"
	if err != nil {
		status = "error"
	}
	o.compactions.WithLabelValues(status).Inc()
}

func (o *PrometheusObserver) OnFlush(d time.Duration, r int, b uint64, err error) {
	o.flushes.Inc()
}

// No-ops for now to keep example simple
func (o *PrometheusObserver) OnGet(d time.Duration, err error)             {}
func (o *PrometheusObserver) OnBuild(d time.Duration, t string, err error) {}
func (o *PrometheusObserver) OnThroughput(n string, b int64)               {}

var (
	targetQPS = flag.Int("qps", 1000, "Target QPS for writes")
	duration  = flag.Duration("duration", 30*time.Second, "Test duration")
)

func main() {
	flag.Parse()

	// 1. Start Prometheus Exporter
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		fmt.Println("Prometheus metrics available at http://localhost:2112/metrics")
		if err := http.ListenAndServe(":2112", nil); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()

	// 2. Initialize Engine
	dir, err := os.MkdirTemp("", "vecgo-obs-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	fmt.Printf("Database dir: %s\n", dir)

	obs := NewPrometheusObserver()
	eng, err := vecgo.Open(context.Background(), vecgo.Local(dir), vecgo.Create(dim, vecgo.MetricL2),
		vecgo.WithMetricsObserver(obs),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	// 3. Generate Load
	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2) // Writer + Reader

	fmt.Printf("Starting load test: %d QPS for %v\n", *targetQPS, *duration)

	// Writer Loop
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(time.Second / time.Duration(*targetQPS))
		defer ticker.Stop()

		vec := make([]float32, dim)

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Random vector using testutil RNG
				rng.FillUniform(vec)

				if _, err := eng.Insert(ctx, vec, nil, nil); err != nil {
					log.Printf("Insert error: %v", err)
				}
			}
		}
	}()

	// Reader Loop (Search)
	go func() {
		defer wg.Done()
		// Search runs as fast as possible but sleeps a tiny bit
		query := make([]float32, dim)

		for {
			select {
			case <-ctx.Done():
				return
			default:
				rng.FillUniform(query)
				_, err := eng.Search(ctx, query, 10)
				if err != nil {
					// Ignore "empty" errors early on
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
	}()

	// Wait for completion
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		cancel()
	}()

	wg.Wait()
	fmt.Println("\nLoad test complete. Creating final snapshot...")

	// Final metrics check
	stats := eng.Stats()
	fmt.Printf("Final Stats: %+v\n", stats)
}
