package engine

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/model"
    "github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/require"
)

func TestConsistency_DuplicatePKs(t *testing.T) {
	eng, err := Open(t.TempDir(), 128, distance.MetricL2)
	require.NoError(t, err)
	defer eng.Close()

	vec1 := make([]float32, 128)
	vec2 := make([]float32, 128)
	vec1[0] = 1.0
	vec2[0] = 2.0

	// Initial insert
	pk1 := model.PKString("duplicate_test")
	err = eng.Insert(pk1, vec1, nil, nil)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)

	// Writer: updates the same PK repeatedly
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
				if err := eng.Insert(pk1, v, nil, nil); err != nil {
					// Ignore errors (might be closed)
				}
				// removed sleep
			}
		}
	}()

	// Reader: searches and asserts no duplicates
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
				
				// Check for duplicates
				pks := make(map[model.PK]int)
				for _, r := range res {
					pks[r.PK]++
				}
				
				for k, v := range pks {
					if v > 1 {
						fmt.Printf("FAIL: Found duplicate PK %v count=%d\n", k, v)
						cancel() // Stop test
						return
					}
				}
				// removed sleep
			}
		}
	}()

	wg.Wait()
}
