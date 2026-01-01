package flat

import (
	"bytes"
	"context"
	"testing"
	"unsafe"

	"github.com/hupe1980/vecgo/index"
)

func ptrInRange(ptr uintptr, base uintptr, n int) bool {
	if n <= 0 {
		return false
	}
	end := base + uintptr(n)
	return ptr >= base && ptr < end
}

func TestMmapLoader_VectorAliasesInputBytes(t *testing.T) {
	ctx := context.Background()

	f, err := New(func(o *Options) {
		o.Dimension = 4
		o.DistanceType = index.DistanceTypeSquaredL2
	})
	if err != nil {
		t.Fatalf("flat.New failed: %v", err)
	}

	id, err := f.Insert(ctx, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	var buf bytes.Buffer
	if _, err := f.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}
	data := buf.Bytes()

	idx, _, err := index.LoadBinaryIndexMmap(data)
	if err != nil {
		t.Fatalf("LoadBinaryIndexMmap failed: %v", err)
	}
	loaded, ok := idx.(*Flat)
	if !ok {
		t.Fatalf("expected *Flat, got %T", idx)
	}

	var found bool
	maxID := loaded.maxID.Load()
	if uint32(id) < maxID && !loaded.deleted.Test(uint32(id)) {
		found = true
	}
	if !found {
		t.Fatalf("expected node id %d to exist", id)
	}
	vec, err := loaded.VectorByID(ctx, id)
	if err != nil {
		t.Fatalf("VectorByID failed: %v", err)
	}
	if len(vec) != 4 {
		t.Fatalf("expected vector len 4, got %d", len(vec))
	}

	base := uintptr(unsafe.Pointer(&data[0]))
	ptr := uintptr(unsafe.Pointer(&vec[0]))
	if !ptrInRange(ptr, base, len(data)) {
		t.Fatalf("expected vector to alias input bytes")
	}
}
