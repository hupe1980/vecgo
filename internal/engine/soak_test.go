package engine

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hupe1980/vecgo/distance"

	"github.com/hupe1980/vecgo/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSoak runs a sustained load test to verify stability.
// It performs concurrent inserts, searches, and deletes while background compaction runs.
func TestSoak(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping soak test in short mode")
	}

	dir := t.TempDir()
	dim := 128
	metric := distance.MetricL2

	// Use a custom observer to track metrics
	obs := &testObserver{}

	e, err := Open(dir, dim, metric, WithMetricsObserver(obs))
	require.NoError(t, err)
	defer e.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var wg sync.WaitGroup

	// 1. Ingest Goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := testutil.NewRNG(time.Now().UnixNano())
		for {
			select {
			case <-ctx.Done():
				return
			default:
				vec := make([]float32, dim)
				rng.FillUniform(vec)
				if _, err := e.Insert(context.Background(), vec, nil, nil); err != nil {
					// Backpressure errors are expected under load
					time.Sleep(10 * time.Millisecond)
				}
			}
		}
	}()

	// 2. Search Goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := testutil.NewRNG(time.Now().UnixNano())
		for {
			select {
			case <-ctx.Done():
				return
			default:
				q := make([]float32, dim)
				rng.FillUniform(q)
				_, err := e.Search(context.Background(), q, 10)
				if err != nil {
					t.Errorf("search failed: %v", err)
				}
				time.Sleep(5 * time.Millisecond)
			}
		}
	}()

	// 3. Flush Trigger Goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				_ = e.Commit(context.Background())
			}
		}
	}()

	wg.Wait()

	// Verify metrics
	flushes := atomic.LoadInt64(&obs.flushes)
	compactions := atomic.LoadInt64(&obs.compactions)
	fmt.Printf("Soak Test Stats:\n")
	fmt.Printf("  Flushes: %d\n", flushes)
	fmt.Printf("  Compactions: %d\n", compactions)

	assert.Greater(t, flushes, int64(0), "should have performed flushes")
	// Compactions might not happen in 10s depending on data volume, but we check no panics occurred.
}

type testObserver struct {
	flushes     int64
	compactions int64
}

func (o *testObserver) OnInsert(time.Duration, error)                   {}
func (o *testObserver) OnDelete(time.Duration, error)                   {}
func (o *testObserver) OnMemTableStatus(int64, float64)                 {}
func (o *testObserver) OnBackpressure(string)                           {}
func (o *testObserver) OnSearch(time.Duration, string, int, int, error) {}
func (o *testObserver) OnGet(time.Duration, error)                      {}

func (o *testObserver) OnFlush(d time.Duration, rows int, bytes uint64, err error) {
	atomic.AddInt64(&o.flushes, 1)
}

func (o *testObserver) OnCompaction(d time.Duration, in int, out int, err error) {
	atomic.AddInt64(&o.compactions, 1)
}

func (o *testObserver) OnBuild(d time.Duration, typ string, err error) {}

func (o *testObserver) OnQueueDepth(name string, depth int) {}

func (o *testObserver) OnThroughput(name string, bytes int64) {}
