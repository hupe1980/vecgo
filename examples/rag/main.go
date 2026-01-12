package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/hupe1980/vecgo"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/testutil"
)

// Document represents a knowledge chunk in our RAG system.
type Document struct {
	ID      uint64
	Content string
	Vector  []float32 // In a real app, this comes from an embedding model (e.g. OpenAI text-embedding-3-small)
}

func main() {
	// 1. Setup: Create a temporary directory for the vector database.
	dir, err := os.MkdirTemp("", "vecgo-rag-example-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir) // Clean up on exit

	fmt.Printf("Initializing Vecgo RAG engine in %s...\n", dir)

	// 2. Initialize the Engine.
	// New unified API: Open(source, opts...)
	// We use L2 distance (Euclidean) and a dimension of 128 for this example.
	// In production, this would match your embedding model (e.g., 1536 for OpenAI).
	const dim = 128
	eng, err := vecgo.Open(dir, vecgo.Create(dim, vecgo.MetricL2))
	if err != nil {
		log.Fatalf("Failed to open engine: %v", err)
	}
	defer eng.Close()

	// 3. Ingest Data (The "Retrieval" Database).
	// We simulate a knowledge base about "Space Exploration".
	docs := []Document{
		{ID: 1, Content: "The Apollo 11 mission landed the first humans on the Moon in 1969."},
		{ID: 2, Content: "Mars is the fourth planet from the Sun and is often called the Red Planet."},
		{ID: 3, Content: "The International Space Station (ISS) is a modular space station in low Earth orbit."},
		{ID: 4, Content: "Voyager 1 is the most distant human-made object from Earth."},
		{ID: 5, Content: "SpaceX was founded by Elon Musk with the goal of reducing space transportation costs."},
	}

	fmt.Printf("Ingesting %d documents...\n", len(docs))
	rng := testutil.NewRNG(time.Now().UnixNano())

	for _, doc := range docs {
		// Simulate embedding generation.
		// In a real application, you would call: embedding := embedder.Embed(doc.Content)
		vec := make([]float32, dim)
		rng.FillUniform(vec)
		doc.Vector = vec

		// Insert into Vecgo.
		// We store the text content as the 'payload'.
		// We could also store metadata (e.g., source, date) in the 3rd argument.
		_, err := eng.Insert(
			doc.Vector,
			metadata.Document{
				"source": metadata.String("wiki-sim"),
				"doc_id": metadata.Int(int64(doc.ID)),
			}, // Metadata
			[]byte(doc.Content), // Payload (The text chunk)
		)
		if err != nil {
			log.Fatalf("Failed to insert doc %d: %v", doc.ID, err)
		}
	}

	// 4. Perform RAG Retrieval.
	// Query: "Tell me about the moon landing."
	fmt.Println("\n--- Performing RAG Retrieval ---")
	queryText := "Tell me about the moon landing."
	fmt.Printf("User Query: %q\n", queryText)

	// Simulate embedding the query.
	// In reality: queryVec := embedder.Embed(queryText)
	queryVec := make([]float32, dim)
	rng.FillUniform(queryVec)

	// Search for the top 3 most relevant chunks.
	// We use vecgo.WithPayload() to fetch the text content in a single round-trip.
	results, err := eng.Search(context.Background(), queryVec, 3, vecgo.WithPayload())
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	// 5. Construct Context for the LLM.
	fmt.Printf("Found %d relevant context chunks:\n", len(results))
	var contextBlock strings.Builder
	for i, res := range results {
		// The payload is returned as []byte.
		content := string(res.Payload)
		fmt.Printf("[%d] (Score: %.4f) %s\n", i+1, res.Score, content)
		fmt.Fprintf(&contextBlock, "- %s\n", content)
	}

	// 6. Generation (Simulated).
	// In a real app, you would send this prompt to an LLM:
	/*
		prompt := fmt.Sprintf(`
		Use the following context to answer the user's question.
		Context:
		%s

		Question: %s
		Answer:`, contextBlock, queryText)

		response := llm.Generate(prompt)
	*/

	fmt.Println("\n--- Simulated LLM Prompt ---")
	fmt.Printf("System: Use the provided context to answer the question.\n")
	fmt.Printf("Context:\n%s\n", contextBlock.String())
	fmt.Printf("User: %s\n", queryText)
	fmt.Println("Assistant: [LLM generates answer based on the above context...]")
}
