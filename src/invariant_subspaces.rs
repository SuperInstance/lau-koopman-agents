//! Koopman invariant subspaces.
//!
//! A subspace V of the observable space is Koopman-invariant if K V ⊆ V.
//! Finding invariant subspaces is crucial for constructing finite-dimensional
//! Koopman approximations that are exact.

use nalgebra::{DMatrix, DVector};
use crate::koopman::KoopmanOperator;

/// A Koopman-invariant subspace.
#[derive(Debug, Clone)]
pub struct InvariantSubspace {
    /// Basis vectors (columns of the matrix).
    pub basis: DMatrix<f64>,
    /// Dimension of the subspace.
    pub dimension: usize,
    /// The restricted Koopman operator on this subspace.
    pub restricted_operator: DMatrix<f64>,
}

impl InvariantSubspace {
    /// Create a new invariant subspace from a basis.
    pub fn new(basis: DMatrix<f64>) -> Self {
        let dim = basis.ncols();
        let restricted_operator = DMatrix::zeros(dim, dim);
        Self { basis, dimension: dim, restricted_operator }
    }

    /// Create an invariant subspace and compute the restricted operator.
    pub fn with_operator(basis: DMatrix<f64>, koopman: &KoopmanOperator) -> Self {
        let dim = basis.ncols();
        // K_restricted = (B^T B)^{-1} B^T K B
        let bt_b = &basis.transpose() * &basis;
        let bt_b_inv = pseudo_inverse_small(&bt_b);
        let restricted = if let Some(inv) = bt_b_inv {
            inv * &basis.transpose() * &koopman.matrix * &basis
        } else {
            DMatrix::zeros(dim, dim)
        };
        Self { basis, dimension: dim, restricted_operator: restricted }
    }

    /// Project a vector onto the subspace.
    pub fn project(&self, v: &DVector<f64>) -> DVector<f64> {
        &self.basis * (&self.basis.transpose() * v)
    }

    /// Check if a vector is approximately in the subspace.
    pub fn contains(&self, v: &DVector<f64>, tol: f64) -> bool {
        let projected = self.project(v);
        (v - projected).norm() < tol * v.norm().max(1.0)
    }

    /// Check invariance: for each basis vector b, K b should be in the subspace.
    pub fn verify_invariance(&self, koopman: &KoopmanOperator, tol: f64) -> bool {
        for i in 0..self.dimension {
            let b = self.basis.column(i).clone_owned();
            let kb = koopman.apply(&b);
            if !self.contains(&kb, tol) {
                return false;
            }
        }
        true
    }

    /// The invariant measure of the subspace (fraction of energy captured).
    pub fn energy_fraction(&self, full_data: &DMatrix<f64>) -> f64 {
        if full_data.ncols() == 0 {
            return 0.0;
        }
        let total_energy: f64 = (0..full_data.ncols())
            .map(|i| full_data.column(i).norm_squared())
            .sum();
        if total_energy < 1e-15 {
            return 0.0;
        }
        let projected_energy: f64 = (0..full_data.ncols())
            .map(|i| self.project(&full_data.column(i).clone_owned()).norm_squared())
            .sum();
        projected_energy / total_energy
    }
}

/// Find an approximately invariant subspace via SVD of the data matrix.
pub fn find_invariant_subspace(
    data: &DMatrix<f64>,
    koopman: &KoopmanOperator,
    dimension: usize,
) -> InvariantSubspace {
    let svd = data.clone().svd(true, false);
    if let Some(u) = svd.u {
        let r = dimension.min(u.ncols());
        let basis = u.columns(0, r).clone_owned();
        InvariantSubspace::with_operator(basis, koopman)
    } else {
        InvariantSubspace::new(DMatrix::zeros(data.nrows(), dimension))
    }
}

/// Check if a set of observables forms an invariant subspace under the dynamics.
pub fn check_invariance(
    observables: &DMatrix<f64>,
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    tol: f64,
) -> bool {
    // Ψ(Y) ≈ K Ψ(X) and Ψ(X) should span the same space as Ψ(Y)
    if x.ncols() == 0 || y.ncols() == 0 {
        return true;
    }
    let psi_x = observables * x;
    let psi_y = observables * y;

    // Check that the column spaces of psi_x and psi_y have the same span
    let svd_x = psi_x.clone().svd(true, false);
    let svd_y = psi_y.clone().svd(true, false);

    if let (Some(u_x), Some(u_y)) = (svd_x.u, svd_y.u) {
        let r = u_x.ncols().min(u_y.ncols());
        // Check that the subspaces are close
        for i in 0..r {
            let col_x = u_x.column(i);
            let col_y = u_y.column(i);
            // Project col_y onto col_x and check distance
            let projection = col_x.dot(&col_y);
            if projection.abs() < 1.0 - tol {
                return false;
            }
        }
        true
    } else {
        false
    }
}

/// Iterate to find a better invariant subspace (power method on the data).
pub fn refine_invariant_subspace(
    initial_basis: &DMatrix<f64>,
    koopman: &KoopmanOperator,
    iterations: usize,
) -> InvariantSubspace {
    let mut basis = initial_basis.clone();
    for _ in 0..iterations {
        // K B
        let kb = &koopman.matrix * &basis;
        // Orthonormalize via QR
        let qr = kb.qr();
        basis = qr.q();
    }
    InvariantSubspace::with_operator(basis, koopman)
}

fn pseudo_inverse_small(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let svd = m.clone().svd(true, true);
    let u = svd.u?;
    let sigma = svd.singular_values;
    let v_t = svd.v_t?;
    let threshold = 1e-10;
    let sigma_inv = DMatrix::from_diagonal(&sigma.map(|s| {
        if s.abs() > threshold { 1.0 / s } else { 0.0 }
    }));
    Some(v_t.transpose() * sigma_inv * u.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_invariant_subspace_creation() {
        let basis = DMatrix::identity(3, 2);
        let is = InvariantSubspace::new(basis);
        assert_eq!(is.dimension, 2);
    }

    #[test]
    fn test_project() {
        let basis = DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]);
        let is = InvariantSubspace::new(basis);
        let v = DVector::from_vec(vec![3.0, 4.0, 5.0]);
        let proj = is.project(&v);
        assert_relative_eq!(proj[0], 3.0);
        assert_relative_eq!(proj[1], 0.0);
        assert_relative_eq!(proj[2], 0.0);
    }

    #[test]
    fn test_contains() {
        let basis = DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]);
        let is = InvariantSubspace::new(basis);
        let v_in = DVector::from_vec(vec![5.0, 0.0, 0.0]);
        let v_out = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        assert!(is.contains(&v_in, 0.1));
        assert!(!is.contains(&v_out, 0.1));
    }

    #[test]
    fn test_invariance_identity() {
        let basis = DMatrix::identity(2, 2);
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        let is = InvariantSubspace::with_operator(basis, &k);
        assert!(is.verify_invariance(&k, 1e-8));
    }

    #[test]
    fn test_invariance_diagonal() {
        let basis = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.8]));
        let is = InvariantSubspace::with_operator(basis, &k);
        assert!(is.verify_invariance(&k, 1e-8));
    }

    #[test]
    fn test_not_invariant() {
        let basis = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.5, 0.3, 0.2, 0.8]));
        let is = InvariantSubspace::with_operator(basis, &k);
        assert!(!is.verify_invariance(&k, 0.01));
    }

    #[test]
    fn test_energy_fraction() {
        let basis = DMatrix::identity(2, 1);
        let is = InvariantSubspace::new(basis);
        let data = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let frac = is.energy_fraction(&data);
        assert_relative_eq!(frac, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_fraction_partial() {
        let basis = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let is = InvariantSubspace::new(basis);
        let data = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let frac = is.energy_fraction(&data);
        assert_relative_eq!(frac, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_find_invariant_subspace() {
        let data = DMatrix::from_row_slice(3, 4, &[
            1.0, 2.0, 3.0, 4.0,
            0.0, 1.0, 2.0, 3.0,
            0.0, 0.0, 0.0, 0.0,
        ]);
        let k = KoopmanOperator::new(DMatrix::identity(3, 3));
        let is = find_invariant_subspace(&data, &k, 2);
        assert_eq!(is.dimension, 2);
    }

    #[test]
    fn test_refine_invariant_subspace() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.0, 0.8]));
        let basis = DMatrix::identity(2, 2);
        let is = refine_invariant_subspace(&basis, &k, 10);
        assert_eq!(is.dimension, 2);
    }
}
