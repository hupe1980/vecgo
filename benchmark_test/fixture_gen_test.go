//go:build genfixtures

// This file generates benchmark fixtures when run with:
//
//	go test -tags=genfixtures -run=TestGenerateFixtures -v ./benchmark_test/...
//
// Fixtures are stored in testdata/fixtures/ and reused across benchmark runs.
package benchmark_test

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// TestGenerateFixtures generates all benchmark fixtures.
// Run with: go test -tags=genfixtures -run=TestGenerateFixtures -v ./benchmark_test/...
func TestGenerateFixtures(t *testing.T) {
	// Ensure testdata directory exists
	if err := os.MkdirAll(fixtureBaseDir, 0755); err != nil {
		t.Fatalf("create fixture base dir: %v", err)
	}

	// Generate quick fixtures by default, all if -short=false
	fixtures := QuickFixtures
	if !testing.Short() {
		fixtures = StandardFixtures
	}

	t.Logf("Generating %d fixtures (use -short=false for all)...", len(fixtures))

	for i, cfg := range fixtures {
		t.Run(cfg.Name, func(t *testing.T) {
			start := time.Now()
			t.Logf("[%d/%d] Generating %s (dim=%d, n=%d, dist=%s)...",
				i+1, len(fixtures), cfg.Name, cfg.Dim, cfg.NumVecs, cfg.Distribution)

			if err := GenerateFixture(cfg); err != nil {
				t.Fatalf("generate fixture: %v", err)
			}

			t.Logf("  Done in %v", time.Since(start).Round(time.Millisecond))
		})
	}
}

// TestGenerateFixture generates a single fixture by name.
// Run with: go test -tags=genfixtures -run=TestGenerateFixture/uniform_128d_50k -v ./benchmark_test/...
func TestGenerateFixture(t *testing.T) {
	for _, cfg := range StandardFixtures {
		t.Run(cfg.Name, func(t *testing.T) {
			// Only run if explicitly requested
			if testing.Short() && cfg.NumVecs > 10_000 {
				t.Skipf("skipping large fixture in short mode")
			}

			start := time.Now()
			t.Logf("Generating %s...", cfg.Name)

			if err := GenerateFixture(cfg); err != nil {
				t.Fatalf("generate fixture: %v", err)
			}

			t.Logf("Done in %v", time.Since(start).Round(time.Millisecond))

			// Verify fixture
			if !FixtureExists(cfg.Name) {
				t.Errorf("fixture %s not created", cfg.Name)
			}

			// Test loading
			data, err := LoadFixtureData(cfg.Name)
			if err != nil {
				t.Errorf("load fixture data: %v", err)
			}
			if len(data.Queries) == 0 {
				t.Error("no queries loaded")
			}
			if len(data.GroundTruth) == 0 {
				t.Error("no ground truth loaded")
			}

			t.Logf("Loaded %d queries, %d selectivity levels", len(data.Queries), len(data.GroundTruth))
		})
	}
}

// TestListFixtures lists all available fixtures.
func TestListFixtures(t *testing.T) {
	t.Log("Standard fixtures:")
	for _, cfg := range StandardFixtures {
		status := "❌ NOT FOUND"
		if FixtureExists(cfg.Name) {
			status = "✅ EXISTS"
		}
		t.Logf("  %s [%s]", cfg.Name, status)
		t.Logf("    dim=%d, n=%d, dist=%s", cfg.Dim, cfg.NumVecs, cfg.Distribution)
	}
}

// TestCleanFixtures removes all fixtures.
func TestCleanFixtures(t *testing.T) {
	if err := os.RemoveAll(fixtureBaseDir); err != nil {
		t.Fatalf("remove fixtures: %v", err)
	}
	t.Log("All fixtures removed")
}

// Helper to print fixture generation command.
func init() {
	if os.Getenv("VECGO_PRINT_FIXTURE_HELP") != "" {
		fmt.Print(`Benchmark Fixture Commands:

  # Generate quick fixtures (CI, ~30s)
  go test -tags=genfixtures -run=TestGenerateFixtures -short -v ./benchmark_test/...

  # Generate all fixtures (~2min)
  go test -tags=genfixtures -run=TestGenerateFixtures -v ./benchmark_test/...
  
  # Generate specific fixture
  go test -tags=genfixtures -run=TestGenerateFixture/uniform_768d_100k -v ./benchmark_test/...

  # List fixture status
  go test -tags=genfixtures -run=TestListFixtures -v ./benchmark_test/...

  # Clean all fixtures
  go test -tags=genfixtures -run=TestCleanFixtures -v ./benchmark_test/...
`)
	}
}
