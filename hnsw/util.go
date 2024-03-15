package hnsw

import "math/rand"

func GenerateRandomVectors(num int, dimensions int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))

	vectors := make([][]float32, num)

	for i := 0; i < num; i++ {
		vectors[i] = make([]float32, dimensions)

		for i2 := 0; i2 < dimensions; i2++ {
			vectors[i][i2] = r.Float32()
		}
	}

	return vectors
}
