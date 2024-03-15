package main

import (
	"fmt"
	"log"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/hnsw"
)

func main() {
	seed := int64(4711)
	dim := 32
	size := 50000
	k := 10

	vg := vecgo.New[int](dim, func(o *vecgo.Options) {
		o.HNSW.M = 32
		//o.HNSW.DistanceFunc = metric.CosineSimilarity
	})

	items := make([]*vecgo.VectorWithData[int], 0, size)
	for i, v := range hnsw.GenerateRandomVectors(size, dim, seed) {
		items = append(items, &vecgo.VectorWithData[int]{
			Vector: v,
			Data:   i,
		})
	}

	query := hnsw.GenerateRandomVectors(1, dim, seed)[0]

	fmt.Println("--- Insert ---")
	fmt.Println("Dimension:", dim)
	fmt.Println("Size:", size)

	start := time.Now()

	for _, item := range items {
		_, err := vg.Insert(item)
		if err != nil {
			log.Fatal(err)
		}
	}
	// if _, err := vg.BatchInsert(items); err != nil {
	// 	log.Fatal(err)
	// }

	end := time.Since(start)

	fmt.Printf("Seconds: %.2f\n\n", end.Seconds())

	vg.PrintStats()
	fmt.Println()

	var (
		err    error
		result []vecgo.SearchResult[int]
	)

	fmt.Println("--- KNN ---")

	start = time.Now()

	result, err = vg.KNNSearch(query, k, func(o *vecgo.KNNSearchOptions) {
		o.EF = 80
	})
	if err != nil {
		log.Fatal(err)
	}

	end = time.Since(start)

	printResult(result)

	fmt.Printf("Seconds: %.8f\n\n", end.Seconds())

	fmt.Println("--- Brute ---")

	start = time.Now()

	result, err = vg.BruteSearch(query, k)
	if err != nil {
		log.Fatal(err)
	}

	end = time.Since(start)

	printResult(result)

	fmt.Printf("Seconds: %.8f\n\n", end.Seconds())
}

func printResult[T any](result []vecgo.SearchResult[T]) {
	for _, r := range result {
		fmt.Printf("ID: %d, Distance: %.2f, Data: %v\n", r.ID, r.Distance, r.Data)
	}
}
