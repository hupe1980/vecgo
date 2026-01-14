package engine

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/require"
)

func TestConsistency_Churn(t *testing.T) {
	eng, err := OpenLocal(context.Background(), t.TempDir(), WithDimension(128), WithMetric(distance.MetricL2))
	require.NoError(t, err)
	defer eng.Close()

	vec1 := make([]float32, 128)
	vec2 := make([]float32, 128)
	vec1[0] = 1.0
	vec2[0] = 2.0

	// Initial insert
	id1, err := eng.Insert(context.Background(), vec1, nil, nil)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	var mu sync.Mutex
	currentID := id1

	// Writer: Deletes old ID, Inserts new ID repeatedly
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			select {
			case <-ctx.Done():
				return
			default:
				// Flip-flop between vec1 and vec2
				v := vec1
				if i%2 == 0 {
					v = vec2
				}

				mu.Lock()
				oldID := currentID
				err := eng.Delete(context.Background(), oldID)
				if err != nil {
					// Ignore valid delete errors
				}
				newID, err := eng.Insert(context.Background(), v, nil, nil)
				if err == nil {
					currentID = newID
				}
				mu.Unlock()
			}
		}
	}()

	// Reader: searches and asserts we don't see ghosts (duplicates)
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			select {
			case <-ctx.Done():
				return
			default:
				// Search k=10
				res, err := eng.Search(context.Background(), vec1, 10)
				if err != nil {
					// Check for closed error?
					continue
				}

				if len(res) > 1 {
					fmt.Printf("FAIL: Found %d results (expected <= 1)\n", len(res))
					for _, r := range res {
						fmt.Printf(" - ID: %d Score: %f\n", r.ID, r.Score)
					}
					cancel()
					t.Error("Consistency violation: Found duplicates (ghosts)")
					return
				}
			}
		}
	}()

	wg.Wait()
}
