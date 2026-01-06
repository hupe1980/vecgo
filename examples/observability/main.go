package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/engine"
	"github.com/hupe1980/vecgo/model"
)

// SimpleObserver implements engine.MetricsObserver and logs metrics.
// It embeds NoopMetricsObserver to avoid implementing all methods.
type SimpleObserver struct {
	engine.NoopMetricsObserver
}

func (o *SimpleObserver) OnInsert(duration time.Duration, err error) {
	if err != nil {
		log.Printf("[METRIC] Insert: failed in %v: %v", duration, err)
	} else {
		// Sample logging to avoid spam
		if duration > 100*time.Microsecond {
			log.Printf("[METRIC] Insert: success in %v", duration)
		}
	}
}

func (o *SimpleObserver) OnSearch(duration time.Duration, segmentType string, k int, results int, err error) {
	log.Printf("[METRIC] Search (%s): k=%d found=%d in %v", segmentType, k, results, duration)
}

func (o *SimpleObserver) OnFlush(duration time.Duration, items int, bytes uint64, err error) {
	log.Printf("[METRIC] Flush: %d items (%d bytes) in %v", items, bytes, duration)
}

func (o *SimpleObserver) OnCompaction(duration time.Duration, dropped int, newSegments int, err error) {
	log.Printf("[METRIC] Compaction: dropped %d items, created %d segments in %v", dropped, newSegments, duration)
}

func main() {
	// dedicated directory for the example
	dir := "./observability_data"
	_ = os.RemoveAll(dir) // Start fresh
	defer os.RemoveAll(dir)

	// Create engine with our observer
	eng, err := engine.Open(dir, 128, distance.MetricL2, engine.WithMetricsObserver(&SimpleObserver{}))
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	log.Println("Engine started with observability...")

	// Perform some inserts
	dim := 128
	count := 1000
	log.Printf("Inserting %d vectors...", count)

	for i := 0; i < count; i++ {
		pk := model.PKUint64(uint64(i))
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}

		if err := eng.Insert(pk, vec, nil, nil); err != nil {
			log.Printf("Insert error: %v", err)
		}
	}

	// Wait a bit or trigger interaction
	time.Sleep(100 * time.Millisecond)

	// Perform search
	log.Println("Performing search...")
	query := make([]float32, dim)
	// fills with random
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}

	res, err := eng.Search(context.Background(), query, 10)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Search found %d results\n", len(res))

	// Get Stats
	stats := eng.Stats()
	fmt.Printf("Engine Stats: %+v\n", stats)

	// Trigger flush manually to see metrics
	log.Println("Triggering Flush...")
	if err := eng.Flush(); err != nil {
		log.Fatal(err)
	}
}
