package conv

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntToUint32(t *testing.T) {
	tests := []struct {
		name    string
		input   int
		want    uint32
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max uint32", math.MaxUint32, math.MaxUint32, false},
		{"invalid negative", -1, 0, true},
		{"invalid too large", math.MaxUint32 + 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Skip "too large" test on 32-bit systems where int is 32-bit
			if tt.name == "invalid too large" && math.MaxInt == math.MaxInt32 {
				t.Skip("skipping overflow test on 32-bit architecture")
			}
			got, err := IntToUint32(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestIntToUint64(t *testing.T) {
	tests := []struct {
		name    string
		input   int
		want    uint64
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max int", math.MaxInt, uint64(math.MaxInt), false},
		{"invalid negative", -1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := IntToUint64(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestIntToUint16(t *testing.T) {
	tests := []struct {
		name    string
		input   int
		want    uint16
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max uint16", math.MaxUint16, math.MaxUint16, false},
		{"invalid negative", -1, 0, true},
		{"invalid too large", math.MaxUint16 + 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := IntToUint16(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestInt64ToUint64(t *testing.T) {
	tests := []struct {
		name    string
		input   int64
		want    uint64
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max int64", math.MaxInt64, uint64(math.MaxInt64), false},
		{"invalid negative", -1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Int64ToUint64(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestUint64ToInt(t *testing.T) {
	tests := []struct {
		name    string
		input   uint64
		want    int
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max int", uint64(math.MaxInt), math.MaxInt, false},
		{"invalid too large", uint64(math.MaxInt) + 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Uint64ToInt(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestUint64ToInt64(t *testing.T) {
	tests := []struct {
		name    string
		input   uint64
		want    int64
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max int64", uint64(math.MaxInt64), math.MaxInt64, false},
		{"invalid too large", uint64(math.MaxInt64) + 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Uint64ToInt64(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestUint64ToUint32(t *testing.T) {
	tests := []struct {
		name    string
		input   uint64
		want    uint32
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max uint32", math.MaxUint32, math.MaxUint32, false},
		{"invalid too large", math.MaxUint32 + 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Uint64ToUint32(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestUint32ToInt32(t *testing.T) {
	tests := []struct {
		name    string
		input   uint32
		want    int32
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max int32", math.MaxInt32, math.MaxInt32, false},
		{"invalid too large", math.MaxInt32 + 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Uint32ToInt32(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestIntToInt32(t *testing.T) {
	tests := []struct {
		name    string
		input   int
		want    int32
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid negative", -123, -123, false},
		{"valid max int32", math.MaxInt32, math.MaxInt32, false},
		{"valid min int32", math.MinInt32, math.MinInt32, false},
		{"invalid too large", math.MaxInt32 + 1, 0, true},
		{"invalid too small", math.MinInt32 - 1, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Skip overflow tests on 32-bit systems where int is 32-bit
			if (tt.name == "invalid too large" || tt.name == "invalid too small") && math.MaxInt == math.MaxInt32 {
				t.Skip("skipping overflow test on 32-bit architecture")
			}
			got, err := IntToInt32(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestUint32ToInt(t *testing.T) {
	tests := []struct {
		name    string
		input   uint32
		want    int
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid max uint32", math.MaxUint32, int(math.MaxUint32), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// On 32-bit systems, MaxUint32 > MaxInt, so this could fail
			if tt.name == "valid max uint32" && uint64(tt.input) > uint64(math.MaxInt) {
				// Expect error on 32-bit systems
				got, err := Uint32ToInt(tt.input)
				assert.Error(t, err)
				assert.Equal(t, 0, got)
				return
			}

			got, err := Uint32ToInt(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestInt64ToInt(t *testing.T) {
	tests := []struct {
		name    string
		input   int64
		want    int
		wantErr bool
	}{
		{"valid zero", 0, 0, false},
		{"valid positive", 123, 123, false},
		{"valid negative", -123, -123, false},
		{"valid max int", int64(math.MaxInt), math.MaxInt, false},
		{"valid min int", int64(math.MinInt), math.MinInt, false},
	}

	// Add overflow tests only if int is 32-bit (where int64 can hold larger values)
	if int64(math.MaxInt) < math.MaxInt64 {
		tests = append(tests, struct {
			name    string
			input   int64
			want    int
			wantErr bool
		}{"invalid too large", int64(math.MaxInt32) + 1, 0, true})
	}
	if int64(math.MinInt) > math.MinInt64 {
		tests = append(tests, struct {
			name    string
			input   int64
			want    int
			wantErr bool
		}{"invalid too small", int64(math.MinInt32) - 1, 0, true})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Int64ToInt(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}
