//! Dynamic Mode Decomposition (DMD).
//!
//! DMD computes a finite-dimensional approximation of the Koopman operator
//! from snapshot data.

use nalgebra::{DMatrix, DVector};
use crate::koopman::KoopmanOperator;

/// DMD result.
#[derive(Debug, Clone)]
pub struct DmdResult {
    pub koopman: KoopmanOperator,
    pub eigenvalues: Vec<num_complex::Complex64>,
    pub modes: DMatrix<f64>,
    pub relative_error: f64,
}

/// Standard DMD: solves min_A ||Y - A X||_F via A = Y X^†
pub fn dmd(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Option<DmdResult> {
    if x.ncols() == 0 || y.ncols() == 0 { return None; }

    let x_pinv = pseudo_inverse(x)?;
    let a = y * x_pinv;

    let svd = x.clone().svd(true, true);
    let _u = svd.u?;
    let sigma = svd.singular_values;
    let v_t = svd.v_t?;

    let threshold = sigma[0] * 1e-10 * (x.nrows().max(x.ncols()) as f64);
    let r = sigma.iter().take_while(|&&s| s > threshold).count().max(1);

    let v_r = v_t.rows(0, r);
    let sigma_r_inv = DMatrix::from_diagonal(&sigma.rows(0, r).map(|s| {
        if s.abs() > 1e-15 { 1.0 / s } else { 0.0 }
    }));
    let modes = y * v_r.transpose() * sigma_r_inv;

    let koopman = KoopmanOperator::new(a);
    let eigenvalues = koopman.eigenvalues();

    let reconstructed = &koopman.matrix * x;
    let error = (y - &reconstructed).norm();
    let y_norm = y.norm();
    let relative_error = if y_norm > 1e-15 { error / y_norm } else { 0.0 };

    Some(DmdResult { koopman, eigenvalues, modes, relative_error })
}

/// DMD from trajectory data.
pub fn dmd_from_trajectory(trajectory: &[DVector<f64>]) -> Option<DmdResult> {
    let n = trajectory.first()?.nrows();
    let m = trajectory.len() - 1;
    if m == 0 { return None; }

    let mut x = DMatrix::zeros(n, m);
    let mut y = DMatrix::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            x[(j, i)] = trajectory[i][j];
            y[(j, i)] = trajectory[i + 1][j];
        }
    }
    dmd(&x, &y)
}

/// Forecast using DMD.
pub fn dmd_forecast(dmd_result: &DmdResult, initial_state: &DVector<f64>, steps: usize) -> Vec<DVector<f64>> {
    let mut result = Vec::with_capacity(steps);
    let mut state = initial_state.clone();
    for _ in 0..steps {
        state = dmd_result.koopman.apply(&state);
        result.push(state.clone());
    }
    result
}

/// Subspace DMD: project onto leading singular vectors first.
pub fn subspace_dmd(x: &DMatrix<f64>, y: &DMatrix<f64>, rank: usize) -> Option<DmdResult> {
    let svd = x.clone().svd(true, true);
    let u = svd.u?;
    let sigma = svd.singular_values;
    let v_t = svd.v_t?;

    let r = rank.min(sigma.len());
    let u_r = u.columns(0, r);
    let v_r = v_t.rows(0, r).transpose();

    let sigma_r_inv = DMatrix::from_diagonal(&sigma.rows(0, r).map(|s| {
        if s.abs() > 1e-15 { 1.0 / s } else { 0.0 }
    }));

    let a_tilde = u_r.transpose() * y * &v_r * sigma_r_inv;
    let a_full = u_r * &a_tilde * u_r.transpose();
    let koopman = KoopmanOperator::new(a_full);

    let eigenvalues = KoopmanOperator::new(a_tilde).eigenvalues();

    let reconstructed = &koopman.matrix * x;
    let error = (y - &reconstructed).norm();
    let y_norm = y.norm();

    Some(DmdResult {
        koopman,
        eigenvalues,
        modes: u_r.clone_owned(),
        relative_error: if y_norm > 1e-15 { error / y_norm } else { 0.0 },
    })
}

/// Moore-Penrose pseudoinverse via SVD.
pub fn pseudo_inverse(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let svd = m.clone().svd(true, true);
    let u = svd.u?;
    let sigma = svd.singular_values;
    let v_t = svd.v_t?;

    let threshold = sigma[0] * 1e-10 * (m.nrows().max(m.ncols()) as f64);
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
    fn test_dmd_identity() {
        let x = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = x.clone();
        let result = dmd(&x, &y).unwrap();
        assert!(result.relative_error < 1e-8);
    }

    #[test]
    fn test_dmd_linear() {
        let x = DMatrix::from_row_slice(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let y = DMatrix::from_row_slice(1, 4, &[0.5, 1.0, 1.5, 2.0]);
        let result = dmd(&x, &y).unwrap();
        assert!(result.relative_error < 1e-8);
        assert_relative_eq!(result.koopman.matrix[(0, 0)], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_dmd_forecast() {
        let x = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        let y = DMatrix::from_row_slice(1, 3, &[2.0, 4.0, 6.0]);
        let result = dmd(&x, &y).unwrap();
        let forecast = dmd_forecast(&result, &DVector::from_vec(vec![5.0]), 3);
        assert_relative_eq!(forecast[0][0], 10.0, epsilon = 0.1);
    }

    #[test]
    fn test_dmd_empty() {
        assert!(dmd(&DMatrix::zeros(2, 0), &DMatrix::zeros(2, 0)).is_none());
    }

    #[test]
    fn test_subspace_dmd() {
        let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = x.scale(0.5);
        let result = subspace_dmd(&x, &y, 1).unwrap();
        assert!(result.relative_error < 0.1);
    }

    #[test]
    fn test_pseudo_inverse_identity() {
        let m = DMatrix::identity(3, 3);
        let pinv = pseudo_inverse(&m).unwrap();
        let product = &m * &pinv;
        for i in 0..3 {
            assert_relative_eq!(product[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pseudo_inverse_tall() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let pinv = pseudo_inverse(&m).unwrap();
        let recovered = &m * &pinv * &m;
        assert_relative_eq!((recovered - m).norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_dmd_2d_system() {
        let a_true = DMatrix::from_row_slice(2, 2, &[0.8, 0.1, 0.0, 0.9]);
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let mut traj = vec![x0.clone()];
        let mut state = x0;
        for _ in 0..20 {
            state = &a_true * &state;
            traj.push(state.clone());
        }
        let result = dmd_from_trajectory(&traj).unwrap();
        assert!(result.relative_error < 0.05);
    }

    #[test]
    fn test_dmd_modes_nonempty() {
        let x = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 2.0]);
        let y = DMatrix::from_row_slice(2, 3, &[0.5, 1.0, 1.5, 0.0, 0.5, 1.0]);
        let result = dmd(&x, &y).unwrap();
        assert!(result.modes.nrows() > 0);
    }

    #[test]
    fn test_dmd_trajectory_too_short() {
        assert!(dmd_from_trajectory(&[DVector::from_vec(vec![1.0, 2.0])]).is_none());
    }
}
