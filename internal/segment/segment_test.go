package segment

import (
	"testing"

	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/assert"
)

func TestSimpleRecordBatch(t *testing.T) {
	// Create sample data
	id1 := model.ID(1)
	id2 := model.ID(2)
	vec1 := []float32{1, 2}
	vec2 := []float32{3, 4}
	meta1 := metadata.Document{"k1": metadata.String("v1")}
	meta2 := metadata.Document{"k2": metadata.Int(2)}
	pay1 := []byte("p1")
	pay2 := []byte("p2")

	batch := &SimpleRecordBatch{}
	batch.IDs = []model.ID{id1, id2}
	batch.Vectors = [][]float32{vec1, vec2}
	batch.Metadatas = []metadata.Document{meta1, meta2}
	batch.Payloads = [][]byte{pay1, pay2}

	assert.Equal(t, 2, batch.RowCount())

	// Test Accessors
	assert.Equal(t, id1, batch.ID(0))
	assert.Equal(t, id2, batch.ID(1))

	assert.Equal(t, vec1, batch.Vector(0))
	assert.Equal(t, vec2, batch.Vector(1))

	assert.Equal(t, meta1, batch.Metadata(0))
	assert.Equal(t, meta2, batch.Metadata(1))

	assert.Equal(t, pay1, batch.Payload(0))
	assert.Equal(t, pay2, batch.Payload(1))
}

func TestSimpleRecordBatch_Partial(t *testing.T) {
	// Only PKs
	batch := &SimpleRecordBatch{
		IDs: []model.ID{model.ID(1)},
	}

	assert.Equal(t, 1, batch.RowCount())
	assert.Equal(t, model.ID(1), batch.ID(0))

	// Other fields nil
	assert.Nil(t, batch.Vector(0))
	assert.Nil(t, batch.Metadata(0))
	assert.Nil(t, batch.Payload(0))
}

func TestStats(t *testing.T) {
	s := FieldStats{
		Min: 0.0,
		Max: 10.0,
	}
	assert.NotNil(t, s)
}
