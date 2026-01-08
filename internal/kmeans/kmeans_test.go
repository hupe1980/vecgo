package kmeans

import (
	"testing"

	"github.com/hupe1980/vecgo/distance"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTrainKMeans(t *testing.T) {
	// 2 clusters: (0,0) and (10,10)
	vecs := []float32{
		0, 0, 0, 1, 1, 0, // near 0,0
		10, 10, 10, 11, 11, 10, // near 10,10
	}
	k := 2
	dim := 2

	centroids, err := TrainKMeans(vecs, dim, k, distance.MetricL2, 100)
	require.NoError(t, err)
	assert.Len(t, centroids, k*dim)

	// Verify assignments
	p1, err := AssignPartition([]float32{0.5, 0.5}, centroids, dim, distance.MetricL2)
	require.NoError(t, err)

	p2, err := AssignPartition([]float32{10.5, 10.5}, centroids, dim, distance.MetricL2)
	require.NoError(t, err)

	assert.NotEqual(t, p1, p2)
}

func TestTrainKMeans_NotEnoughVectors(t *testing.T) {
	vecs := []float32{0, 0}
	centroids, err := TrainKMeans(vecs, 2, 2, distance.MetricL2, 10)
	require.NoError(t, err)
	assert.Nil(t, centroids)
}

func TestTrainKMeans_Error(t *testing.T) {
	_, err := TrainKMeans([]float32{0, 0}, 2, 1, distance.Metric(999), 10)
	assert.Error(t, err)
}

func TestFindClosestCentroids(t *testing.T) {
	centroids := []float32{
		0, 0, // 0
		10, 10, // 1
		20, 20, // 2
	}
	dim := 2

	// Query close to 0,0
	res, err := FindClosestCentroids([]float32{1, 1}, centroids, dim, 2, distance.MetricL2)
	require.NoError(t, err)
	assert.Len(t, res, 2)
	assert.Equal(t, 0, res[0])
	assert.Equal(t, 1, res[1])

	// Query close to 20,20
	res, err = FindClosestCentroids([]float32{19, 19}, centroids, dim, 1, distance.MetricL2)
	require.NoError(t, err)
	assert.Len(t, res, 1)
	assert.Equal(t, 2, res[0])

	// Error case (invalid metric)
	_, err = FindClosestCentroids([]float32{0, 0}, centroids, dim, 1, distance.Metric(999))
	assert.Error(t, err)
}

func TestAssignPartition_Error(t *testing.T) {
	_, err := AssignPartition([]float32{0, 0}, []float32{0, 0}, 2, distance.Metric(999))
	assert.Error(t, err)
}
