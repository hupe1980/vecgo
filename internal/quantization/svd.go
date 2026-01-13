package quantization

import (
	"fmt"
	"math"
)

// svd computes the Singular Value Decomposition of matrix A = U * Sigma * V^T.
// This is a simplified implementation using one-sided Jacobi rotations, suitable for small matrices (e.g. < 64x64).
// It returns U, Sigma (as a slice), and V.
// Note: This implementation modifies the input matrix 'a' (it becomes U * Sigma).
// To preserve 'a', pass a copy.
func svd(a [][]float32) ([][]float32, []float32, [][]float32) {
	m := len(a)
	if m == 0 {
		return nil, nil, nil
	}
	n := len(a[0])

	// Initialize V as identity matrix
	v := make([][]float32, n)
	for i := range v {
		v[i] = make([]float32, n)
		v[i][i] = 1.0
	}

	// U will be accumulated in 'a'
	u := a

	// One-sided Jacobi SVD
	// Iterate until convergence
	const tol = 1e-5
	const maxIter = 100

	for range maxIter {
		if !performJacobiIterations(u, v, m, n, tol) {
			break
		}
	}

	// Extract singular values (norms of columns of U)
	sigma := make([]float32, n)
	for j := range n {
		sum := float32(0)
		for i := range m {
			sum += u[i][j] * u[i][j]
		}
		sigma[j] = float32(math.Sqrt(float64(sum)))

		// Normalize columns of U
		if sigma[j] > 1e-10 {
			inv := 1.0 / sigma[j]
			for i := range m {
				u[i][j] *= inv
			}
		}
	}

	return u, sigma, v
}

func performJacobiIterations(u, v [][]float32, m, n int, tol float64) bool {
	changed := false
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			// Compute dot products
			alpha := float32(0) // column i . column i
			beta := float32(0)  // column j . column j
			gamma := float32(0) // column i . column j

			for k := range m {
				alpha += u[k][i] * u[k][i]
				beta += u[k][j] * u[k][j]
				gamma += u[k][i] * u[k][j]
			}

			// Check for underflow
			if alpha < 1e-12 || beta < 1e-12 {
				continue
			}

			// Check for orthogonality
			if math.Abs(float64(gamma)) < tol*math.Sqrt(float64(alpha*beta)) {
				continue
			}

			changed = true
			applyRotation(u, v, m, n, i, j, alpha, beta, gamma)
		}
	}
	return changed
}

func applyRotation(u, v [][]float32, m, n, i, j int, alpha, beta, gamma float32) {
	// Compute Jacobi rotation
	zeta := (beta - alpha) / (2 * gamma)
	var t float32
	if zeta > 0 {
		t = 1 / (zeta + float32(math.Sqrt(float64(1+zeta*zeta))))
	} else {
		t = -1 / (-zeta + float32(math.Sqrt(float64(1+zeta*zeta))))
	}
	c := 1 / float32(math.Sqrt(float64(1+t*t)))
	s := c * t

	// Apply rotation to U (columns i and j)
	for k := range m {
		t1 := u[k][i]
		t2 := u[k][j]
		u[k][i] = c*t1 - s*t2
		u[k][j] = s*t1 + c*t2
	}

	// Apply rotation to V (columns i and j)
	for k := range n {
		t1 := v[k][i]
		t2 := v[k][j]
		v[k][i] = c*t1 - s*t2
		v[k][j] = s*t1 + c*t2
	}
}

// computeProcrustesRotation computes the optimal rotation matrix R such that || A*R - B ||_F is minimized.
// This is equivalent to maximizing Tr(R^T * M) where M = A^T * B.
// The solution is R = U * V^T where M = U * Sigma * V^T.
// Returns R.
func computeProcrustesRotation(mMatrix [][]float32) ([][]float32, error) {
	if len(mMatrix) == 0 || len(mMatrix) != len(mMatrix[0]) {
		return nil, fmt.Errorf("procrustes requires square matrix")
	}

	u, sigma, v := svd(mMatrix)
	dim := len(mMatrix)

	// Find index of smallest singular value
	minSigmaIdx := 0
	minSigma := sigma[0]
	for i := 1; i < dim; i++ {
		if sigma[i] < minSigma {
			minSigma = sigma[i]
			minSigmaIdx = i
		}
	}

	// Compute R = U * V^T
	// R_ij = sum_k (U_ik * V^T_kj) = sum_k (U_ik * V_jk)
	r := make([][]float32, dim)
	for i := range r {
		r[i] = make([]float32, dim)
		for j := range dim {
			sum := float32(0)
			for k := range dim {
				sum += u[i][k] * v[j][k]
			}
			r[i][j] = sum
		}
	}

	// Check determinant of R
	det := determinant(r)

	if det < 0 {
		// Reflection detected. Flip the column of U corresponding to the smallest singular value.
		// This minimizes the error introduced by forcing a rotation.
		for i := range dim {
			u[i][minSigmaIdx] *= -1
		}

		// Recompute R = U * V^T
		for i := range r {
			for j := range dim {
				sum := float32(0)
				for k := range dim {
					sum += u[i][k] * v[j][k]
				}
				r[i][j] = sum
			}
		}
	}

	return r, nil
}

func determinant(matrix [][]float32) float32 {
	n := len(matrix)
	if n == 0 {
		return 0
	}
	// Clone matrix to avoid modification
	temp := make([][]float32, n)
	for i := range temp {
		temp[i] = make([]float32, n)
		copy(temp[i], matrix[i])
	}

	det := float32(1.0)
	for i := range n {
		pivot := i
		for j := i + 1; j < n; j++ {
			if math.Abs(float64(temp[j][i])) > math.Abs(float64(temp[pivot][i])) {
				pivot = j
			}
		}

		if pivot != i {
			temp[i], temp[pivot] = temp[pivot], temp[i]
			det *= -1
		}

		if temp[i][i] == 0 {
			return 0
		}

		det *= temp[i][i]

		for j := i + 1; j < n; j++ {
			factor := temp[j][i] / temp[i][i]
			for k := i + 1; k < n; k++ {
				temp[j][k] -= factor * temp[i][k]
			}
		}
	}
	return det
}
