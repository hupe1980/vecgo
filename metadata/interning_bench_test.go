package metadata_test

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/hupe1980/vecgo/metadata"
)

// NaiveValue mimics the old metadata.Value without interning
type NaiveValue struct {
	Kind metadata.Kind
	S    string
	I64  int64
	F64  float64
	B    bool
	A    []NaiveValue
}

// NaiveDocument mimics the old metadata.Document (map[string]Value)
type NaiveDocument map[string]NaiveValue

func TestMemoryUsageComparison(t *testing.T) {
	// Configuration
	const numDocs = 100_000
	const uniqueKeys = 10   // Only 10 unique keys repeated
	const uniqueValues = 50 // Only 50 unique string values repeated
	const fieldsPerDoc = 5

	// Generate pool of strings
	keys := make([]string, uniqueKeys)
	for i := range keys {
		keys[i] = fmt.Sprintf("key_%d", i)
	}
	values := make([]string, uniqueValues)
	for i := range values {
		values[i] = fmt.Sprintf("value_long_string_payload_for_testing_memory_usage_%d", i)
	}

	// Helper to force GC and get heap usage
	getHeapUsage := func() uint64 {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		return m.Alloc
	}

	t.Run("Naive_Implementation", func(t *testing.T) {
		startMem := getHeapUsage()

		docs := make([]NaiveDocument, numDocs)
		for i := 0; i < numDocs; i++ {
			doc := make(NaiveDocument, fieldsPerDoc)
			for j := 0; j < fieldsPerDoc; j++ {
				k := string([]byte(keys[(i+j)%uniqueKeys]))     // Force allocation
				v := string([]byte(values[(i+j)%uniqueValues])) // Force allocation
				doc[k] = NaiveValue{
					Kind: metadata.KindString,
					S:    v, // Direct string storage (duplicate headers)
				}
			}
			docs[i] = doc
		}

		endMem := getHeapUsage()
		usageMB := float64(endMem-startMem) / 1024 / 1024
		t.Logf("Naive Implementation Memory: %.2f MB", usageMB)

		// Keep docs alive
		runtime.KeepAlive(docs)
	})

	t.Run("Interned_Implementation", func(t *testing.T) {
		startMem := getHeapUsage()

		docs := make([]metadata.Document, numDocs)
		for i := 0; i < numDocs; i++ {
			doc := make(metadata.Document, fieldsPerDoc)
			for j := 0; j < fieldsPerDoc; j++ {
				k := string([]byte(keys[(i+j)%uniqueKeys]))     // Force allocation
				v := string([]byte(values[(i+j)%uniqueValues])) // Force allocation
				doc[k] = metadata.String(v)
			}
			docs[i] = doc
		}

		endMem := getHeapUsage()
		usageMB := float64(endMem-startMem) / 1024 / 1024
		t.Logf("Interned Implementation Memory: %.2f MB", usageMB)

		// Keep docs alive
		runtime.KeepAlive(docs)
	})

	t.Run("UnifiedIndex_Storage", func(t *testing.T) {
		// This tests the full internal storage (interned keys + interned values)
		startMem := getHeapUsage()

		idx := metadata.NewUnifiedIndex()
		for i := 0; i < numDocs; i++ {
			doc := make(metadata.Document, fieldsPerDoc)
			for j := 0; j < fieldsPerDoc; j++ {
				k := string([]byte(keys[(i+j)%uniqueKeys]))     // Force allocation
				v := string([]byte(values[(i+j)%uniqueValues])) // Force allocation
				doc[k] = metadata.String(v)
			}
			idx.Set(uint64(i), doc)
		}

		endMem := getHeapUsage()
		usageMB := float64(endMem-startMem) / 1024 / 1024
		t.Logf("UnifiedIndex (Full Interning) Memory: %.2f MB", usageMB)

		runtime.KeepAlive(idx)
	})
}
