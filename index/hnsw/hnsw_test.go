package hnsw

import (
	"fmt"
	"log"
	"testing"

	"github.com/hupe1980/vecgo/util"
	"github.com/stretchr/testify/assert"
)

type TestCases struct {
	VectorSize int
	VectorDim  int

	M         int
	EF        int
	Heuristic bool
	K         int

	Precision float64
}

func TestNew(t *testing.T) {
	h := New(func(o *Options) {
		o.M = 8
		o.EF = 200
	})

	assert.Equal(t, 8, h.opts.M)
	assert.Equal(t, 8, h.mmax)
	assert.Equal(t, 16, h.mmax0)
	assert.Equal(t, 200, h.opts.EF)
}

func TestValidateInsertSearch(t *testing.T) {
	tests := []TestCases{
		{
			VectorSize: 1000,
			VectorDim:  16,
			M:          8,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 1000,
			VectorDim:  16,
			M:          8,
			EF:         200,
			Heuristic:  false,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 1000,
			VectorDim:  1024,
			M:          12,
			EF:         200,
			Heuristic:  true,
			Precision:  0.98,
			K:          10,
		},
		{
			VectorSize: 10000,
			VectorDim:  16,
			M:          16,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 10000,
			VectorDim:  32,
			M:          16,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 1000,
			VectorDim:  16,
			M:          8,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 1000,
			VectorDim:  16,
			M:          8,
			EF:         200,
			Heuristic:  false,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 1000,
			VectorDim:  1024,
			M:          12,
			EF:         200,
			Heuristic:  true,
			Precision:  0.98,
			K:          10,
		},
		{
			VectorSize: 10000,
			VectorDim:  16,
			M:          16,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
		{
			VectorSize: 10000,
			VectorDim:  32,
			M:          16,
			EF:         200,
			Heuristic:  true,
			Precision:  0.99,
			K:          10,
		},
	}

	for _, tc := range tests {
		testname := fmt.Sprintf("Vec=%d,Dim=%d,Heuristic=%t,M=%d,Precision=%f", tc.VectorSize, tc.VectorDim, tc.Heuristic, tc.M, tc.Precision)

		t.Run(testname, func(t *testing.T) {
			rng := util.NewRNG(4711)

			vecs := rng.GenerateRandomVectors(tc.VectorSize, tc.VectorDim)

			assert.Equal(t, tc.VectorSize, len(vecs))
			assert.Equal(t, tc.VectorDim, len(vecs[0]))

			h := New(func(o *Options) {
				o.M = tc.M
				o.EF = tc.EF
				o.Heuristic = tc.Heuristic
			})

			for i := 0; i < len(vecs); i++ {
				id, err := h.Insert(vecs[i])
				assert.GreaterOrEqual(t, id, uint32(0))
				assert.Nil(t, err)
			}

			groundResults := make([][]uint32, len(vecs))

			for i := 0; i < len(vecs); i++ {
				bestCandidatesBrute, _ := h.BruteSearch(vecs[i], tc.K, func(id uint32) bool { return true })

				groundResults[i] = make([]uint32, tc.K)

				for i2, item := range bestCandidatesBrute {
					groundResults[i][i2] = item.ID
				}
			}

			hitSuccess := 0
			totalSearch := 0

			for i := 0; i < len(vecs); i++ {
				bestCandidates, err := h.KNNSearch(vecs[i], tc.K, tc.EF, func(id uint32) bool { return true })
				if err != nil {
					log.Fatal(err)
				}

				for _, item := range bestCandidates {
					if len(bestCandidates) == 0 {
						fmt.Println("No matches")
						break
					}

					totalSearch++

					for k := tc.K - 1; k >= 0; k-- {
						if item.ID == groundResults[i][k] {
							hitSuccess++
						}
					}
				}
			}

			precision := float64(hitSuccess) / (float64(len(vecs)) * float64(tc.K))

			fmt.Printf("Precision => %f\n", precision)
			assert.GreaterOrEqual(t, precision, tc.Precision)
		})
	}
}
