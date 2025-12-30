package vecgo_test

import (
	"context"
	"testing"

	"github.com/hupe1980/vecgo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCRUDOperations(t *testing.T) {
	testCases := []struct {
		name    string
		factory func(t *testing.T) *vecgo.Vecgo[string]
	}{
		{
			name: "Flat",
			factory: func(t *testing.T) *vecgo.Vecgo[string] {
				vg, err := vecgo.Flat[string](3).SquaredL2().Build()
				require.NoError(t, err)
				return vg
			},
		},
		{
			name: "HNSW",
			factory: func(t *testing.T) *vecgo.Vecgo[string] {
				vg, err := vecgo.HNSW[string](3).SquaredL2().Build()
				require.NoError(t, err)
				return vg
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("Delete", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert vectors
				id1, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   "first",
				})
				require.NoError(t, err)

				id2, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{4.0, 5.0, 6.0},
					Data:   "second",
				})
				require.NoError(t, err)

				id3, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{7.0, 8.0, 9.0},
					Data:   "third",
				})
				require.NoError(t, err)

				// Verify all exist
				data, err := vg.Get(id1)
				require.NoError(t, err)
				assert.Equal(t, "first", data)

				data, err = vg.Get(id2)
				require.NoError(t, err)
				assert.Equal(t, "second", data)

				data, err = vg.Get(id3)
				require.NoError(t, err)
				assert.Equal(t, "third", data)

				// Delete middle item
				err = vg.Delete(context.Background(), id2)
				require.NoError(t, err)

				// Verify deleted item is not found
				_, err = vg.Get(id2)
				assert.ErrorIs(t, err, vecgo.ErrNotFound)

				// Verify others still exist
				data, err = vg.Get(id1)
				require.NoError(t, err)
				assert.Equal(t, "first", data)

				data, err = vg.Get(id3)
				require.NoError(t, err)
				assert.Equal(t, "third", data)

				// Search should not return deleted item
				results, err := vg.BruteSearch(context.Background(), []float32{4.0, 5.0, 6.0}, 3)
				require.NoError(t, err)
				require.Len(t, results, 2)

				// Verify results don't contain deleted item
				for _, r := range results {
					assert.NotEqual(t, id2, r.ID)
				}
			})

			t.Run("DeleteNonExistent", func(t *testing.T) {
				vg := tc.factory(t)

				// Try to delete non-existent ID
				err := vg.Delete(context.Background(), 999)
				assert.Error(t, err)
			})

			t.Run("Update", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert vector
				id, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   "original",
				})
				require.NoError(t, err)

				// Verify original value
				data, err := vg.Get(id)
				require.NoError(t, err)
				assert.Equal(t, "original", data)

				// Update
				err = vg.Update(context.Background(), id, vecgo.VectorWithData[string]{
					Vector: []float32{10.0, 20.0, 30.0},
					Data:   "updated",
				})
				require.NoError(t, err)

				// Verify updated value
				data, err = vg.Get(id)
				require.NoError(t, err)
				assert.Equal(t, "updated", data)

				// Search with new vector should find it
				results, err := vg.BruteSearch(context.Background(), []float32{10.0, 20.0, 30.0}, 1)
				require.NoError(t, err)
				require.Len(t, results, 1)
				assert.Equal(t, id, results[0].ID)
				assert.Equal(t, "updated", results[0].Data)

				// Search with old vector should not find it as best match
				results, err = vg.BruteSearch(context.Background(), []float32{1.0, 2.0, 3.0}, 1)
				require.NoError(t, err)
				require.Len(t, results, 1)
				assert.Equal(t, id, results[0].ID) // Still returns it (only item) but distance should be higher
			})

			t.Run("UpdateNonExistent", func(t *testing.T) {
				vg := tc.factory(t)

				// Try to update non-existent ID
				err := vg.Update(context.Background(), 999, vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   "test",
				})
				assert.Error(t, err)
			})

			t.Run("UpdateDeleted", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert item
				id, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   "test",
				})
				require.NoError(t, err)

				// Delete the item
				err = vg.Delete(context.Background(), id)
				require.NoError(t, err)

				// Try to update deleted item
				err = vg.Update(context.Background(), id, vecgo.VectorWithData[string]{
					Vector: []float32{10.0, 20.0, 30.0},
					Data:   "updated",
				})
				assert.Error(t, err)
			})

			t.Run("UpdateWithDifferentDimensions", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert vector with 3 dimensions
				id, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   "original",
				})
				require.NoError(t, err)

				// Try to update with different dimensions
				err = vg.Update(context.Background(), id, vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0, 4.0}, // 4 dimensions
					Data:   "updated",
				})
				assert.Error(t, err)
			})

			t.Run("InsertAfterDelete", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert multiple items
				id1, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   "first",
				})
				require.NoError(t, err)

				id2, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{10.0, 11.0, 12.0},
					Data:   "second",
				})
				require.NoError(t, err)

				// Delete second item
				err = vg.Delete(context.Background(), id2)
				require.NoError(t, err)

				// Insert new item after deletion
				id3, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
					Vector: []float32{4.0, 5.0, 6.0},
					Data:   "third",
				})
				require.NoError(t, err)

				// Verify new item exists
				data, err := vg.Get(id3)
				require.NoError(t, err)
				assert.Equal(t, "third", data)

				// With ID reuse: id3 should equal id2 (reused from free list)
				// The "second" data should be replaced with "third" at that ID
				assert.Equal(t, id2, id3, "ID should be reused from free list")

				// Verify id2 now contains "third" (not "second")
				data, err = vg.Get(id2)
				require.NoError(t, err)
				assert.Equal(t, "third", data, "Reused ID should have new data")

				// First item should still exist unchanged
				data, err = vg.Get(id1)
				require.NoError(t, err)
				assert.Equal(t, "first", data)
			})

			t.Run("MultipleDeletesAndUpdates", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert multiple vectors
				ids := make([]uint64, 0, 10)
				for i := 0; i < 10; i++ {
					id, err := vg.Insert(context.Background(), vecgo.VectorWithData[string]{
						Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
						Data:   string(rune('a' + i)),
					})
					require.NoError(t, err)
					ids = append(ids, id)
				}

				// Delete even indices
				for i := 0; i < 10; i += 2 {
					err := vg.Delete(context.Background(), ids[i])
					require.NoError(t, err)
				}

				// Update odd indices
				for i := 1; i < 10; i += 2 {
					err := vg.Update(context.Background(), ids[i], vecgo.VectorWithData[string]{
						Vector: []float32{float32(i * 10), float32(i*10 + 1), float32(i*10 + 2)},
						Data:   string(rune('A' + i)),
					})
					require.NoError(t, err)
				}

				// Verify deleted IDs are gone
				for i := 0; i < 10; i += 2 {
					_, err := vg.Get(ids[i])
					assert.ErrorIs(t, err, vecgo.ErrNotFound)
				}

				// Verify odd IDs are updated
				for i := 1; i < 10; i += 2 {
					data, err := vg.Get(ids[i])
					require.NoError(t, err)
					assert.Equal(t, string(rune('A'+i)), data)
				}
			})
		})
	}
}

func TestConcurrentCRUDOperations(t *testing.T) {
	testCases := []struct {
		name    string
		factory func(t *testing.T) *vecgo.Vecgo[int]
	}{
		{
			name: "Flat",
			factory: func(t *testing.T) *vecgo.Vecgo[int] {
				vg, err := vecgo.Flat[int](3).SquaredL2().Build()
				require.NoError(t, err)
				return vg
			},
		},
		{
			name: "HNSW",
			factory: func(t *testing.T) *vecgo.Vecgo[int] {
				vg, err := vecgo.HNSW[int](3).SquaredL2().Build()
				require.NoError(t, err)
				return vg
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("ConcurrentInserts", func(t *testing.T) {
				vg := tc.factory(t)
				n := 100

				errs := make(chan error, n)
				ids := make(chan uint64, n)

				for i := 0; i < n; i++ {
					go func(val int) {
						id, err := vg.Insert(context.Background(), vecgo.VectorWithData[int]{
							Vector: []float32{float32(val), float32(val + 1), float32(val + 2)},
							Data:   val,
						})
						if err != nil {
							errs <- err
							return
						}
						ids <- id
					}(i)
				}

				// Collect results
				idSlice := make([]uint64, 0, n)
				for i := 0; i < n; i++ {
					select {
					case err := <-errs:
						t.Fatalf("Insert failed: %v", err)
					case id := <-ids:
						idSlice = append(idSlice, id)
					}
				}

				// Verify all inserts succeeded
				assert.Len(t, idSlice, n)
			})

			t.Run("ConcurrentReads", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert test data
				id, err := vg.Insert(context.Background(), vecgo.VectorWithData[int]{
					Vector: []float32{1.0, 2.0, 3.0},
					Data:   42,
				})
				require.NoError(t, err)

				// Concurrent reads
				n := 100
				errs := make(chan error, n)

				for i := 0; i < n; i++ {
					go func() {
						data, err := vg.Get(id)
						if err != nil || data != 42 {
							errs <- err
							return
						}
						errs <- nil
					}()
				}

				// Check results
				for i := 0; i < n; i++ {
					err := <-errs
					assert.NoError(t, err)
				}
			})

			t.Run("ConcurrentUpdates", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert initial data
				ids := make([]uint64, 10)
				for i := 0; i < 10; i++ {
					id, err := vg.Insert(context.Background(), vecgo.VectorWithData[int]{
						Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
						Data:   i,
					})
					require.NoError(t, err)
					ids[i] = id
				}

				// Concurrent updates
				n := 100
				errs := make(chan error, n)

				for i := 0; i < n; i++ {
					go func(val int) {
						idx := val % len(ids)
						err := vg.Update(context.Background(), ids[idx], vecgo.VectorWithData[int]{
							Vector: []float32{float32(val), float32(val + 1), float32(val + 2)},
							Data:   val,
						})
						errs <- err
					}(i)
				}

				// Check results
				for i := 0; i < n; i++ {
					err := <-errs
					assert.NoError(t, err)
				}
			})

			t.Run("ConcurrentDeletes", func(t *testing.T) {
				vg := tc.factory(t)

				// Insert initial data
				ids := make([]uint64, 10)
				for i := 0; i < 10; i++ {
					id, err := vg.Insert(context.Background(), vecgo.VectorWithData[int]{
						Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
						Data:   i,
					})
					require.NoError(t, err)
					ids[i] = id
				}

				// Concurrent deletes - all nodes can now be deleted
				deleteIds := ids[1:] // Keep first ID for validation
				errs := make(chan error, len(deleteIds))

				for i := 0; i < len(deleteIds); i++ {
					go func(id uint64) {
						errs <- vg.Delete(context.Background(), id)
					}(deleteIds[i])
				}

				// All deletes should succeed
				for i := 0; i < len(deleteIds); i++ {
					err := <-errs
					assert.NoError(t, err)
				}

				// Verify all deleted items are gone
				for i := 1; i < len(ids); i++ {
					_, err := vg.Get(ids[i])
					assert.ErrorIs(t, err, vecgo.ErrNotFound)
				}
			})
		})
	}
}
