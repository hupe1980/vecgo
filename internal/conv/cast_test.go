//go:build amd64 || arm64

package conv

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntToUint32(t *testing.T) {
	t.Run("valid zero", func(t *testing.T) {
		got, err := IntToUint32(0)
		assert.NoError(t, err)
		assert.Equal(t, uint32(0), got)
	})

	t.Run("valid positive", func(t *testing.T) {
		got, err := IntToUint32(123)
		assert.NoError(t, err)
		assert.Equal(t, uint32(123), got)
	})

	t.Run("invalid negative", func(t *testing.T) {
		_, err := IntToUint32(-1)
		assert.Error(t, err)
	})

	t.Run("valid max int32", func(t *testing.T) {
		got, err := IntToUint32(math.MaxInt32)
		assert.NoError(t, err)
		assert.Equal(t, uint32(math.MaxInt32), got)
	})
}

func TestIntToUint64(t *testing.T) {
	t.Run("valid zero", func(t *testing.T) {
		got, err := IntToUint64(0)
		assert.NoError(t, err)
		assert.Equal(t, uint64(0), got)
	})

	t.Run("valid positive", func(t *testing.T) {
		got, err := IntToUint64(123)
		assert.NoError(t, err)
		assert.Equal(t, uint64(123), got)
	})

	t.Run("valid max int", func(t *testing.T) {
		got, err := IntToUint64(math.MaxInt)
		assert.NoError(t, err)
		assert.Equal(t, uint64(math.MaxInt), got)
	})

	t.Run("invalid negative", func(t *testing.T) {
		_, err := IntToUint64(-1)
		assert.Error(t, err)
	})
}

func TestUint64ToInt(t *testing.T) {
	t.Run("valid zero", func(t *testing.T) {
		got, err := Uint64ToInt(0)
		assert.NoError(t, err)
		assert.Equal(t, 0, got)
	})

	t.Run("valid positive", func(t *testing.T) {
		got, err := Uint64ToInt(123)
		assert.NoError(t, err)
		assert.Equal(t, 123, got)
	})

	t.Run("valid max int", func(t *testing.T) {
		got, err := Uint64ToInt(uint64(math.MaxInt))
		assert.NoError(t, err)
		assert.Equal(t, math.MaxInt, got)
	})

	t.Run("invalid too large", func(t *testing.T) {
		_, err := Uint64ToInt(uint64(math.MaxInt) + 1)
		assert.Error(t, err)
	})
}

func TestUint64ToUint32(t *testing.T) {
	t.Run("valid zero", func(t *testing.T) {
		got, err := Uint64ToUint32(0)
		assert.NoError(t, err)
		assert.Equal(t, uint32(0), got)
	})

	t.Run("valid positive", func(t *testing.T) {
		got, err := Uint64ToUint32(123)
		assert.NoError(t, err)
		assert.Equal(t, uint32(123), got)
	})

	t.Run("valid max uint32", func(t *testing.T) {
		got, err := Uint64ToUint32(math.MaxUint32)
		assert.NoError(t, err)
		assert.Equal(t, uint32(math.MaxUint32), got)
	})

	t.Run("invalid too large", func(t *testing.T) {
		_, err := Uint64ToUint32(math.MaxUint32 + 1)
		assert.Error(t, err)
	})
}

func TestUint32ToInt(t *testing.T) {
	t.Run("valid zero", func(t *testing.T) {
		got, err := Uint32ToInt(0)
		assert.NoError(t, err)
		assert.Equal(t, 0, got)
	})

	t.Run("valid positive", func(t *testing.T) {
		got, err := Uint32ToInt(123)
		assert.NoError(t, err)
		assert.Equal(t, 123, got)
	})

	t.Run("max uint32", func(t *testing.T) {
		got, err := Uint32ToInt(math.MaxUint32)
		// On 64-bit (amd64/arm64), MaxUint32 fits in int
		// This test runs on supported 64-bit platforms
		assert.NoError(t, err)
		assert.Equal(t, int(math.MaxUint32), got)
	})
}
