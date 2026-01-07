package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
)

func main() {
	dir := "./data-modern"
	os.RemoveAll(dir)
	defer os.RemoveAll(dir)

	// 1. Setup Structured Logger
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// 2. Define Schema
	schema := metadata.Schema{
		"category": metadata.FieldTypeString,
		"price":    metadata.FieldTypeFloat,
		"tags":     metadata.FieldTypeArray,
	}

	// 3. Open Engine with Modern Options
	eng, err := vecgo.Open(dir, 4, vecgo.MetricL2,
		vecgo.WithLogger(logger),
		vecgo.WithSchema(schema),
	)
	if err != nil {
		logger.Error("Failed to open engine", "error", err)
		os.Exit(1)
	}
	defer eng.Close()

	// 4. Insert Data (Generic PKs + Typed Metadata)
	logger.Info("Inserting data...")
	err = eng.Insert(
		vecgo.PKString("prod-1"),
		[]float32{1.0, 0.0, 0.0, 0.0},
		metadata.Document{
			"category": metadata.String("electronics"),
			"price":    metadata.Float(99.99),
			"tags":     metadata.Strings([]string{"gadget", "sale"}),
		},
		[]byte(`{"desc": "A cool gadget"}`),
	)
	if err != nil {
		logger.Error("Insert failed", "error", err)
	}

	err = eng.Insert(
		vecgo.PKString("prod-2"),
		[]float32{0.0, 1.0, 0.0, 0.0},
		metadata.Document{
			"category": metadata.String("books"),
			"price":    metadata.Float(19.50),
			"tags":     metadata.Strings([]string{"fiction"}),
		},
		nil,
	)
	if err != nil {
		logger.Error("Insert failed", "error", err)
	}

	// 5. Scan using Go 1.23 iterators
	logger.Info("Scanning all records...")
	for record, err := range eng.Scan(context.Background()) {
		if err != nil {
			logger.Error("Scan error", "error", err)
			break
		}
		fmt.Printf("Record: PK=%v, Vector=%v, Meta=%v\n", record.PK, record.Vector, record.Metadata)
	}

	// 6. Search
	results, err := eng.Search(context.Background(), []float32{1.0, 0.0, 0.0, 0.0}, 1,
		vecgo.WithMetadata(),
	)
	if err != nil {
		logger.Error("Search failed", "error", err)
	}

	for _, res := range results {
		fmt.Printf("Match: PK=%v Score=%f\n", res.PK, res.Score)
	}
}
