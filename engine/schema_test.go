package engine

import (
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/hupe1980/vecgo/metadata"
	"github.com/hupe1980/vecgo/model"
	"github.com/stretchr/testify/require"
)

func TestSchemaValidation(t *testing.T) {
	schema := metadata.Schema{
		"age":    metadata.FieldTypeInt,
		"name":   metadata.FieldTypeString,
		"active": metadata.FieldTypeBool,
		"score":  metadata.FieldTypeFloat,
		"tags":   metadata.FieldTypeArray,
	}

	e, err := Open(t.TempDir(), 128, distance.MetricL2, WithSchema(schema))
	require.NoError(t, err)
	defer e.Close()

	vec := make([]float32, 128)

	tests := []struct {
		name    string
		md      map[string]any
		wantErr bool
	}{
		{
			name: "valid metadata",
			md: map[string]any{
				"age":    25,
				"name":   "John",
				"active": true,
				"score":  0.95,
				"tags":   []string{"a", "b"},
			},
			wantErr: false,
		},
		{
			name: "invalid int (string)",
			md: map[string]any{
				"age": "25",
			},
			wantErr: true,
		},
		{
			name: "invalid string (int)",
			md: map[string]any{
				"name": 123,
			},
			wantErr: true,
		},
		{
			name: "invalid bool (int)",
			md: map[string]any{
				"active": 1,
			},
			wantErr: true,
		},
		{
			name: "valid float (int allowed)",
			md: map[string]any{
				"score": 1,
			},
			wantErr: false,
		},
		{
			name: "unknown field (allowed)",
			md: map[string]any{
				"extra": "data",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := e.Insert(model.PKUint64(1), vec, tt.md, nil)
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestBatchInsertSchemaValidation(t *testing.T) {
	schema := metadata.Schema{
		"age": metadata.FieldTypeInt,
	}

	e, err := Open(t.TempDir(), 128, distance.MetricL2, WithSchema(schema))
	require.NoError(t, err)
	defer e.Close()

	vec := make([]float32, 128)

	// Valid batch
	err = e.BatchInsert([]model.Record{
		{PK: model.PKUint64(1), Vector: vec, Metadata: map[string]any{"age": 20}},
		{PK: model.PKUint64(2), Vector: vec, Metadata: map[string]any{"age": 30}},
	})
	require.NoError(t, err)

	// Invalid batch
	err = e.BatchInsert([]model.Record{
		{PK: model.PKUint64(3), Vector: vec, Metadata: map[string]any{"age": 20}},
		{PK: model.PKUint64(4), Vector: vec, Metadata: map[string]any{"age": "invalid"}},
	})
	require.Error(t, err)
}
