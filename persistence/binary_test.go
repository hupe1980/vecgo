package persistence

import (
	"bytes"
	"io"
	"os"
	"testing"
)

func TestBinaryFormat_WriteRead(t *testing.T) {
	// Test vector data
	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
	}

	// Write to buffer
	var buf bytes.Buffer
	writer := NewBinaryIndexWriter(&buf)

	header := &FileHeader{
		VectorCount: uint64(len(vectors)),
		Dimension:   4,
		IndexType:   IndexTypeFlat,
	}

	if err := writer.WriteHeader(header); err != nil {
		t.Fatalf("WriteHeader failed: %v", err)
	}

	for _, vec := range vectors {
		if err := writer.WriteFloat32Slice(vec); err != nil {
			t.Fatalf("WriteFloat32Slice failed: %v", err)
		}
	}

	// Read back
	reader := NewBinaryIndexReader(&buf)

	readHeader, err := reader.ReadHeader()
	if err != nil {
		t.Fatalf("ReadHeader failed: %v", err)
	}

	if readHeader.VectorCount != header.VectorCount {
		t.Errorf("VectorCount mismatch: got %d, want %d", readHeader.VectorCount, header.VectorCount)
	}

	if readHeader.Dimension != header.Dimension {
		t.Errorf("Dimension mismatch: got %d, want %d", readHeader.Dimension, header.Dimension)
	}

	for i := 0; i < len(vectors); i++ {
		vec, err := reader.ReadFloat32Slice(int(header.Dimension))
		if err != nil {
			t.Fatalf("ReadFloat32Slice failed: %v", err)
		}

		for j, v := range vec {
			if v != vectors[i][j] {
				t.Errorf("Vector %d mismatch at index %d: got %f, want %f", i, j, v, vectors[i][j])
			}
		}
	}
}

func TestSaveLoadFile(t *testing.T) {
	tmpfile := "test_index.bin"
	defer os.Remove(tmpfile)

	// Test data
	testVectors := []float32{1.1, 2.2, 3.3, 4.4}

	// Save
	err := SaveToFile(tmpfile, func(w io.Writer) error {
		writer := NewBinaryIndexWriter(w)
		header := &FileHeader{
			VectorCount: 1,
			Dimension:   4,
			IndexType:   IndexTypeFlat,
		}
		if err := writer.WriteHeader(header); err != nil {
			return err
		}
		return writer.WriteFloat32Slice(testVectors)
	})

	if err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Load
	var loadedVectors []float32
	err = LoadFromFile(tmpfile, func(r io.Reader) error {
		reader := NewBinaryIndexReader(r)
		_, err := reader.ReadHeader()
		if err != nil {
			return err
		}
		loadedVectors, err = reader.ReadFloat32Slice(4)
		return err
	})

	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	// Verify
	for i, v := range loadedVectors {
		if v != testVectors[i] {
			t.Errorf("Vector mismatch at %d: got %f, want %f", i, v, testVectors[i])
		}
	}
}

func TestHNSWMetadata_WriteRead(t *testing.T) {
	var buf bytes.Buffer

	original := &HNSWMetadata{
		MaxLayers:    5,
		M:            16,
		Ml:           1.0 / 0.69314718056,
		EntryPoint:   42,
		DistanceFunc: 1,
	}

	if err := WriteHNSWMetadata(&buf, original); err != nil {
		t.Fatalf("WriteHNSWMetadata failed: %v", err)
	}

	loaded, err := ReadHNSWMetadata(&buf)
	if err != nil {
		t.Fatalf("ReadHNSWMetadata failed: %v", err)
	}

	if loaded.MaxLayers != original.MaxLayers {
		t.Errorf("MaxLayers mismatch: got %d, want %d", loaded.MaxLayers, original.MaxLayers)
	}
	if loaded.M != original.M {
		t.Errorf("M mismatch: got %d, want %d", loaded.M, original.M)
	}
	if loaded.EntryPoint != original.EntryPoint {
		t.Errorf("EntryPoint mismatch: got %d, want %d", loaded.EntryPoint, original.EntryPoint)
	}
}

func TestConnections_WriteRead(t *testing.T) {
	var buf bytes.Buffer

	connections := [][]uint64{
		{1, 2, 3},
		{4, 5},
		{6, 7, 8, 9},
	}

	if err := WriteConnections(&buf, connections); err != nil {
		t.Fatalf("WriteConnections failed: %v", err)
	}

	loaded, err := ReadConnections(&buf, 3)
	if err != nil {
		t.Fatalf("ReadConnections failed: %v", err)
	}

	if len(loaded) != len(connections) {
		t.Fatalf("Layer count mismatch: got %d, want %d", len(loaded), len(connections))
	}

	for i, layer := range connections {
		if len(loaded[i]) != len(layer) {
			t.Errorf("Layer %d length mismatch: got %d, want %d", i, len(loaded[i]), len(layer))
			continue
		}
		for j, conn := range layer {
			if loaded[i][j] != conn {
				t.Errorf("Connection mismatch at layer %d, index %d: got %d, want %d", i, j, loaded[i][j], conn)
			}
		}
	}
}

func BenchmarkWriteFloat32Slice(b *testing.B) {
	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i)
	}

	var buf bytes.Buffer
	writer := NewBinaryIndexWriter(&buf)

	b.ResetTimer()
	for b.Loop() {
		buf.Reset()
		writer.WriteFloat32Slice(vec)
	}
}

func BenchmarkReadFloat32Slice(b *testing.B) {
	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i)
	}

	var buf bytes.Buffer
	writer := NewBinaryIndexWriter(&buf)
	writer.WriteFloat32Slice(vec)

	data := buf.Bytes()

	b.ResetTimer()
	for b.Loop() {
		reader := NewBinaryIndexReader(bytes.NewReader(data))
		reader.ReadFloat32Slice(128)
	}
}
