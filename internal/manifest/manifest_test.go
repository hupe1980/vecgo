package manifest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManifestVersioning(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(nil, dir)

	// 1. Save a manifest
	m := &Manifest{
		ID: 0,
	}
	err := store.Save(m)
	require.NoError(t, err)

	// 2. Load it
	loaded, err := store.Load()
	require.NoError(t, err)
	assert.Equal(t, CurrentVersion, loaded.Version)

	// 3. Corrupt version
	// Find the manifest file
	currentContent, err := os.ReadFile(filepath.Join(dir, CurrentFileName))
	require.NoError(t, err)
	manifestPath := filepath.Join(dir, string(currentContent))

	// Read JSON
	data, err := os.ReadFile(manifestPath)
	require.NoError(t, err)

	var raw map[string]interface{}
	err = json.Unmarshal(data, &raw)
	require.NoError(t, err)

	// Modify version
	raw["version"] = 999
	newData, err := json.Marshal(raw)
	require.NoError(t, err)

	// Write back
	err = os.WriteFile(manifestPath, newData, 0644)
	require.NoError(t, err)

	// 4. Load again - should fail
	_, err = store.Load()
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrIncompatibleVersion)
}
