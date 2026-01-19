package vectorstore

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/hupe1980/vecgo/model"
	"github.com/hupe1980/vecgo/testutil"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name string
		dim  int
		want int
	}{
		{"positive dim", 128, 128},
		{"zero dim", 0, 1},
		{"negative dim", -10, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := New(tt.dim, nil)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if s.Dimension() != tt.want {
				t.Errorf("Dimension() = %v, want %v", s.Dimension(), tt.want)
			}
			if s.Count() != 0 {
				t.Errorf("Count() = %v, want 0", s.Count())
			}
			if s.LiveCount() != 0 {
				t.Errorf("LiveCount() = %v, want 0", s.LiveCount())
			}
		})
	}
}

func TestAppend(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	vec1 := []float32{1.0, 2.0, 3.0}
	id1, err := s.Append(vec1)
	if err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if id1 != 0 {
		t.Errorf("first id = %v, want 0", id1)
	}

	vec2 := []float32{4.0, 5.0, 6.0}
	id2, err := s.Append(vec2)
	if err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if id2 != 1 {
		t.Errorf("second id = %v, want 1", id2)
	}

	if s.Count() != 2 {
		t.Errorf("Count() = %v, want 2", s.Count())
	}
	if s.LiveCount() != 2 {
		t.Errorf("LiveCount() = %v, want 2", s.LiveCount())
	}

	got1, ok := s.GetVector(0)
	if !ok {
		t.Error("GetVector(0) returned false")
	}
	if !equalFloat32(got1, vec1) {
		t.Errorf("GetVector(0) = %v, want %v", got1, vec1)
	}

	got2, ok := s.GetVector(1)
	if !ok {
		t.Error("GetVector(1) returned false")
	}
	if !equalFloat32(got2, vec2) {
		t.Errorf("GetVector(1) = %v, want %v", got2, vec2)
	}
}

func TestAppend_WrongDimension(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	_, err = s.Append([]float32{1.0, 2.0})
	if err == nil {
		t.Error("Append() with wrong dimension should return error")
	}
}

func TestSetVector(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	vec := []float32{1.0, 2.0, 3.0}
	if err := s.SetVector(0, vec); err != nil {
		t.Fatalf("SetVector() error = %v", err)
	}

	got, ok := s.GetVector(0)
	if !ok {
		t.Error("GetVector(0) returned false")
	}
	if !equalFloat32(got, vec) {
		t.Errorf("GetVector(0) = %v, want %v", got, vec)
	}

	vec5 := []float32{7.0, 8.0, 9.0}
	if err := s.SetVector(5, vec5); err != nil {
		t.Fatalf("SetVector(5) error = %v", err)
	}

	got0, ok := s.GetVector(0)
	if !ok {
		t.Error("GetVector(0) returned false after SetVector(5)")
	}
	if !equalFloat32(got0, vec) {
		t.Errorf("GetVector(0) = %v, want %v", got0, vec)
	}

	got5, ok := s.GetVector(5)
	if !ok {
		t.Error("GetVector(5) returned false")
	}
	if !equalFloat32(got5, vec5) {
		t.Errorf("GetVector(5) = %v, want %v", got5, vec5)
	}
}

func TestDelete(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	s.Append([]float32{1.0, 2.0, 3.0})
	s.Append([]float32{4.0, 5.0, 6.0})
	s.Append([]float32{7.0, 8.0, 9.0})

	if err := s.DeleteVector(1); err != nil {
		t.Fatalf("DeleteVector() error = %v", err)
	}

	if s.Count() != 3 {
		t.Errorf("Count() = %v, want 3", s.Count())
	}
	if s.LiveCount() != 2 {
		t.Errorf("LiveCount() = %v, want 2", s.LiveCount())
	}

	if _, ok := s.GetVector(1); ok {
		t.Error("GetVector(1) should return false for deleted vector")
	}

	if _, ok := s.GetVector(0); !ok {
		t.Error("GetVector(0) returned false")
	}
	if _, ok := s.GetVector(2); !ok {
		t.Error("GetVector(2) returned false")
	}

	if !s.IsDeleted(1) {
		t.Error("IsDeleted(1) should be true")
	}
	if s.IsDeleted(0) {
		t.Error("IsDeleted(0) should be false")
	}
}

func TestDelete_OutOfBounds(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	s.Append([]float32{1.0, 2.0, 3.0})

	if err := s.DeleteVector(10); err == nil {
		t.Error("DeleteVector() with out of bounds ID should return error")
	}
}

func TestCompact(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	vec0 := []float32{1.0, 2.0, 3.0}
	vec1 := []float32{4.0, 5.0, 6.0}
	vec2 := []float32{7.0, 8.0, 9.0}
	vec3 := []float32{10.0, 11.0, 12.0}

	s.Append(vec0)
	s.Append(vec1)
	s.Append(vec2)
	s.Append(vec3)

	s.DeleteVector(1)
	s.DeleteVector(2)

	if s.LiveCount() != 2 {
		t.Errorf("LiveCount() = %v, want 2", s.LiveCount())
	}

	idMap, err := s.Compact()
	if err != nil {
		t.Fatalf("Compact() failed: %v", err)
	}

	if idMap[0] != 0 {
		t.Errorf("idMap[0] = %v, want 0", idMap[0])
	}
	if idMap[3] != 1 {
		t.Errorf("idMap[3] = %v, want 1", idMap[3])
	}
	if _, ok := idMap[1]; ok {
		t.Error("idMap should not contain deleted ID 1")
	}
	if _, ok := idMap[2]; ok {
		t.Error("idMap should not contain deleted ID 2")
	}

	if s.Count() != 2 {
		t.Errorf("Count() = %v, want 2", s.Count())
	}
	if s.LiveCount() != 2 {
		t.Errorf("LiveCount() = %v, want 2", s.LiveCount())
	}

	got0, ok := s.GetVector(0)
	if !ok {
		t.Error("GetVector(0) returned false")
	}
	if !equalFloat32(got0, vec0) {
		t.Errorf("GetVector(0) = %v, want %v", got0, vec0)
	}

	got1, ok := s.GetVector(1)
	if !ok {
		t.Error("GetVector(1) returned false")
	}
	if !equalFloat32(got1, vec3) {
		t.Errorf("GetVector(1) = %v, want %v", got1, vec3)
	}
}

func TestCompact_NothingToCompact(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	s.Append([]float32{1.0, 2.0, 3.0})
	s.Append([]float32{4.0, 5.0, 6.0})

	idMap, err := s.Compact()
	if err != nil {
		t.Fatalf("Compact() failed: %v", err)
	}
	if idMap != nil {
		t.Errorf("Compact() with no deletions should return nil, got %v", idMap)
	}
}

func TestIterate(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	s.Append([]float32{1.0, 2.0, 3.0})
	s.Append([]float32{4.0, 5.0, 6.0})
	s.Append([]float32{7.0, 8.0, 9.0})
	s.DeleteVector(1)

	var visited []model.RowID
	s.Iterate(func(id model.RowID, vec []float32) bool {
		visited = append(visited, id)
		return true
	})

	if len(visited) != 2 {
		t.Errorf("Iterate visited %v IDs, want 2", len(visited))
	}
	if visited[0] != 0 || visited[1] != 2 {
		t.Errorf("Iterate visited %v, want [0 2]", visited)
	}
}

func TestIterate_EarlyStop(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	for i := 0; i < 10; i++ {
		s.Append([]float32{float32(i), float32(i), float32(i)})
	}

	count := 0
	s.Iterate(func(id model.RowID, vec []float32) bool {
		count++
		return count < 3
	})

	if count != 3 {
		t.Errorf("Iterate count = %v, want 3", count)
	}
}

func TestWriteRead(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	s.Append([]float32{1.0, 2.0, 3.0})
	s.Append([]float32{4.0, 5.0, 6.0})
	s.Append([]float32{7.0, 8.0, 9.0})
	s.DeleteVector(1)

	var buf bytes.Buffer
	_, err = s.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo() error = %v", err)
	}

	s2, err := New(0, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	_, err = s2.ReadFrom(&buf)
	if err != nil {
		t.Fatalf("ReadFrom() error = %v", err)
	}

	if s2.Dimension() != 3 {
		t.Errorf("Dimension() = %v, want 3", s2.Dimension())
	}
	if s2.Count() != 3 {
		t.Errorf("Count() = %v, want 3", s2.Count())
	}
	if s2.LiveCount() != 2 {
		t.Errorf("LiveCount() = %v, want 2", s2.LiveCount())
	}

	got0, ok := s2.GetVector(0)
	if !ok {
		t.Error("GetVector(0) returned false")
	}
	if !equalFloat32(got0, []float32{1.0, 2.0, 3.0}) {
		t.Errorf("GetVector(0) = %v, want [1 2 3]", got0)
	}

	if _, ok := s2.GetVector(1); ok {
		t.Error("GetVector(1) should return false for deleted vector")
	}

	got2, ok := s2.GetVector(2)
	if !ok {
		t.Error("GetVector(2) returned false")
	}
	if !equalFloat32(got2, []float32{7.0, 8.0, 9.0}) {
		t.Errorf("GetVector(2) = %v, want [7 8 9]", got2)
	}
}

func TestMmap(t *testing.T) {
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "test.col")

	s, err := New(128, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	vecs := make([][]float32, 100)
	for i := range vecs {
		vecs[i] = make([]float32, 128)
		for j := range vecs[i] {
			vecs[i][j] = float32(i*128 + j)
		}
		s.Append(vecs[i])
	}
	s.DeleteVector(50)

	f, err := os.Create(filename)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	_, err = s.WriteTo(f)
	f.Close()
	if err != nil {
		t.Fatalf("WriteTo() error = %v", err)
	}

	ms, closer, err := OpenMmap(filename)
	if err != nil {
		t.Fatalf("OpenMmap() error = %v", err)
	}
	defer closer.Close()

	if ms.Dimension() != 128 {
		t.Errorf("Dimension() = %v, want 128", ms.Dimension())
	}
	if ms.Count() != 100 {
		t.Errorf("Count() = %v, want 100", ms.Count())
	}
	if ms.LiveCount() != 99 {
		t.Errorf("LiveCount() = %v, want 99", ms.LiveCount())
	}

	for i := 0; i < 100; i++ {
		if i == 50 {
			if _, ok := ms.GetVector(model.RowID(i)); ok {
				t.Errorf("GetVector(%d) should return false for deleted vector", i)
			}
			continue
		}
		got, ok := ms.GetVector(model.RowID(i))
		if !ok {
			t.Errorf("GetVector(%d) returned false", i)
			continue
		}
		if !equalFloat32(got, vecs[i]) {
			t.Errorf("GetVector(%d) mismatch", i)
		}
	}

	if err := ms.SetVector(0, vecs[0]); err == nil {
		t.Error("SetVector() on mmap store should return error")
	}
	if err := ms.DeleteVector(0); err == nil {
		t.Error("DeleteVector() on mmap store should return error")
	}
}

func TestConcurrentRead(t *testing.T) {
	s, err := New(64, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 64)
		for j := range vec {
			vec[j] = float32(i*64 + j)
		}
		s.Append(vec)
	}

	rng := testutil.NewRNG(0)

	var wg sync.WaitGroup
	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				id := uint64(rng.Intn(1000))
				s.GetVector(model.RowID(id))
			}
		}()
	}
	wg.Wait()
}

func TestRawData(t *testing.T) {
	s, err := New(3, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	s.Append([]float32{1.0, 2.0, 3.0})
	s.Append([]float32{4.0, 5.0, 6.0})

	raw, _ := s.RawData()
	if len(raw) != 6 {
		t.Errorf("RawData() len = %v, want 6", len(raw))
	}
	expected := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	if !equalFloat32(raw, expected) {
		t.Errorf("RawData() = %v, want %v", raw, expected)
	}
}

func TestLargeStore(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large store test in short mode")
	}

	dim := 128
	count := 100000
	s, err := New(dim, nil)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	vec := make([]float32, dim)
	for i := 0; i < count; i++ {
		for j := range vec {
			vec[j] = float32(i*dim + j)
		}
		if _, err := s.Append(vec); err != nil {
			t.Fatalf("Append() error at %d: %v", i, err)
		}
	}

	if s.Count() != uint64(count) {
		t.Errorf("Count() = %v, want %v", s.Count(), count)
	}

	rng := testutil.NewRNG(0)

	for i := 0; i < 100; i++ {
		id := uint64(rng.Intn(count))
		got, ok := s.GetVector(model.RowID(id))
		if !ok {
			t.Errorf("GetVector(%d) returned false", id)
			continue
		}
		if got[0] != float32(int(id)*dim) {
			t.Errorf("GetVector(%d)[0] = %v, want %v", id, got[0], float32(int(id)*dim))
		}
	}
}

func equalFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkAppend(b *testing.B) {
	dims := []int{64, 128, 256, 512, 1024}
	for _, dim := range dims {
		b.Run("dim="+string(rune('0'+dim/100)), func(b *testing.B) {
			s, err := New(dim, nil)
			if err != nil {
				b.Fatalf("New() error = %v", err)
			}
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = float32(i)
			}
			b.ResetTimer()
			for b.Loop() {
				s.Append(vec)
			}
		})
	}
}

func BenchmarkGetVector(b *testing.B) {
	dim := 128
	s, err := New(dim, nil)
	if err != nil {
		b.Fatalf("New() error = %v", err)
	}
	for i := 0; i < 10000; i++ {
		vec := make([]float32, dim)
		s.Append(vec)
	}

	b.ResetTimer()
	var i int
	for b.Loop() {
		s.GetVector(model.RowID(i % 10000))
		i++
	}
}

func BenchmarkIterate(b *testing.B) {
	dim := 128
	s, err := New(dim, nil)
	if err != nil {
		b.Fatalf("New() error = %v", err)
	}
	for i := 0; i < 10000; i++ {
		vec := make([]float32, dim)
		s.Append(vec)
	}

	b.ResetTimer()
	for b.Loop() {
		s.Iterate(func(id model.RowID, vec []float32) bool {
			return true
		})
	}
}
