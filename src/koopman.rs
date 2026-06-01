//! Koopman operator theory.
//!
//! For a dynamical system x_{k+1} = F(x_k), the Koopman operator K acts on
//! observable functions g as: (K g)(x) = g(F(x)).
//! K is linear but infinite-dimensional, operating on the space of observables.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A discrete-time dynamical system x_{k+1} = F(x_k).
pub trait DynamicalSystem: Clone {
    /// The state dimension.
    fn state_dim(&self) -> usize;
    /// Advance the state by one step: x_{k+1} = F(x_k).
    fn step(&self, state: &DVector<f64>) -> DVector<f64>;
}

/// A simple linear dynamical system x_{k+1} = A x_k.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSystem {
    /// State transition matrix A.
    pub a: DMatrix<f64>,
}

impl LinearSystem {
    pub fn new(a: DMatrix<f64>) -> Self {
        Self { a }
    }
}

impl DynamicalSystem for LinearSystem {
    fn state_dim(&self) -> usize {
        self.a.nrows()
    }
    fn step(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.a * state
    }
}

/// A nonlinear dynamical system defined by a closure.
#[derive(Clone)]
pub struct FnSystem {
    pub dim: usize,
    pub f: fn(&DVector<f64>) -> DVector<f64>,
}

impl FnSystem {
    pub fn new(dim: usize, f: fn(&DVector<f64>) -> DVector<f64>) -> Self {
        Self { dim, f }
    }
}

impl DynamicalSystem for FnSystem {
    fn state_dim(&self) -> usize { self.dim }
    fn step(&self, state: &DVector<f64>) -> DVector<f64> {
        (self.f)(state)
    }
}

/// The Koopman operator (finite-dimensional approximation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoopmanOperator {
    /// Finite-dimensional approximation of K.
    pub matrix: DMatrix<f64>,
    /// Number of observables.
    pub num_observables: usize,
}

impl KoopmanOperator {
    /// Create a Koopman operator from its matrix representation.
    pub fn new(matrix: DMatrix<f64>) -> Self {
        let n = matrix.nrows();
        Self { matrix, num_observables: n }
    }

    /// Apply the Koopman operator to an observable vector.
    pub fn apply(&self, observable: &DVector<f64>) -> DVector<f64> {
        &self.matrix * observable
    }

    /// Compute eigenvalues using the Schur decomposition.
    pub fn eigenvalues(&self) -> Vec<Complex64> {
        let n = self.matrix.nrows();
        if n == 0 {
            return vec![];
        }
        let eigen = self.matrix.complex_eigenvalues();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(Complex64::new(eigen[i].re, eigen[i].im));
        }
        result
    }

    /// Check if the operator is stable (all eigenvalues inside unit circle).
    pub fn is_stable(&self) -> bool {
        self.eigenvalues().iter().all(|λ| λ.norm() <= 1.0 + 1e-10)
    }

    /// The spectral radius (largest eigenvalue magnitude).
    pub fn spectral_radius(&self) -> f64 {
        self.eigenvalues().iter().map(|λ| λ.norm()).fold(0.0_f64, f64::max)
    }

    /// Power iteration to estimate dominant eigenvalue.
    pub fn dominant_eigenvalue(&self, max_iter: usize) -> Complex64 {
        let n = self.num_observables;
        if n == 0 {
            return Complex64::new(0.0, 0.0);
        }
        let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
        let mut eigenvalue = 0.0_f64;
        for _ in 0..max_iter {
            let av = &self.matrix * &v;
            let new_eigenvalue = v.dot(&av);
            let norm = av.norm();
            if norm < 1e-15 { break; }
            v = av / norm;
            eigenvalue = new_eigenvalue;
        }
        Complex64::new(eigenvalue, 0.0)
    }

    /// Compute Koopman modes.
    pub fn modes(&self, initial_observables: &DVector<f64>) -> Vec<(Complex64, DVector<f64>)> {
        let eigenvalues = self.eigenvalues();
        let n = self.num_observables;
        eigenvalues.into_iter().map(|λ| {
            let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
            for _ in 0..50 {
                let av = &self.matrix * &v;
                let norm = av.norm();
                if norm < 1e-15 { break; }
                v = av / norm;
            }
            let amplitude = v.dot(initial_observables);
            (λ, v.scale(amplitude))
        }).collect()
    }
}

/// Generate a trajectory from a dynamical system.
pub fn generate_trajectory<S: DynamicalSystem>(
    system: &S,
    initial_state: &DVector<f64>,
    num_steps: usize,
) -> Vec<DVector<f64>> {
    let mut trajectory = Vec::with_capacity(num_steps + 1);
    trajectory.push(initial_state.clone());
    let mut state = initial_state.clone();
    for _ in 0..num_steps {
        state = system.step(&state);
        trajectory.push(state.clone());
    }
    trajectory
}

/// Stack trajectory data into snapshot matrices X, Y where Y = F(X).
pub fn stack_snapshots(trajectory: &[DVector<f64>]) -> (DMatrix<f64>, DMatrix<f64>) {
    if trajectory.len() < 2 {
        let n = trajectory.first().map(|v| v.nrows()).unwrap_or(0);
        return (DMatrix::zeros(n, 0), DMatrix::zeros(n, 0));
    }
    let n = trajectory[0].nrows();
    let m = trajectory.len() - 1;
    let mut x = DMatrix::zeros(n, m);
    let mut y = DMatrix::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            x[(j, i)] = trajectory[i][j];
            y[(j, i)] = trajectory[i + 1][j];
        }
    }
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_system_step() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.5]);
        let sys = LinearSystem::new(a);
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = sys.step(&x);
        assert_relative_eq!(y[0], 1.0);
        assert_relative_eq!(y[1], 1.0);
    }

    #[test]
    fn test_linear_system_dim() {
        let sys = LinearSystem::new(DMatrix::identity(3, 3));
        assert_eq!(sys.state_dim(), 3);
    }

    #[test]
    fn test_koopman_identity() {
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        let v = DVector::from_vec(vec![3.0, 4.0]);
        let r = k.apply(&v);
        assert_relative_eq!(r[0], 3.0);
        assert_relative_eq!(r[1], 4.0);
    }

    #[test]
    fn test_koopman_stable_identity() {
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        assert!(k.is_stable());
    }

    #[test]
    fn test_koopman_unstable() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 2.0));
        assert!(!k.is_stable());
    }

    #[test]
    fn test_koopman_spectral_radius() {
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        assert_relative_eq!(k.spectral_radius(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_koopman_dominant_eigenvalue() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 3.0));
        let λ = k.dominant_eigenvalue(100);
        assert_relative_eq!(λ.re, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_generate_trajectory_identity() {
        let sys = LinearSystem::new(DMatrix::identity(2, 2));
        let x0 = DVector::from_vec(vec![1.0, 2.0]);
        let traj = generate_trajectory(&sys, &x0, 5);
        assert_eq!(traj.len(), 6);
        for t in &traj {
            assert_relative_eq!(t[0], 1.0);
            assert_relative_eq!(t[1], 2.0);
        }
    }

    #[test]
    fn test_generate_trajectory_decay() {
        let sys = LinearSystem::new(DMatrix::from_row_slice(1, 1, &[0.5]));
        let x0 = DVector::from_vec(vec![8.0]);
        let traj = generate_trajectory(&sys, &x0, 3);
        assert_relative_eq!(traj[0][0], 8.0);
        assert_relative_eq!(traj[1][0], 4.0);
        assert_relative_eq!(traj[2][0], 2.0);
        assert_relative_eq!(traj[3][0], 1.0);
    }

    #[test]
    fn test_stack_snapshots() {
        let sys = LinearSystem::new(DMatrix::identity(2, 2));
        let x0 = DVector::from_vec(vec![1.0, 2.0]);
        let traj = generate_trajectory(&sys, &x0, 2);
        let (x, y) = stack_snapshots(&traj);
        assert_eq!(x.ncols(), 2);
        assert_eq!(y.ncols(), 2);
    }

    #[test]
    fn test_stack_snapshots_empty() {
        let traj: Vec<DVector<f64>> = vec![];
        let (x, y) = stack_snapshots(&traj);
        assert_eq!(x.ncols(), 0);
    }

    #[test]
    fn test_koopman_modes() {
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        let init = DVector::from_vec(vec![1.0, 0.0]);
        let modes = k.modes(&init);
        assert!(!modes.is_empty());
    }

    #[test]
    fn test_fn_system() {
        let sys = FnSystem::new(2, |x| DVector::from_vec(vec![x[0] * 0.5, x[1] * 0.5]));
        let x = DVector::from_vec(vec![4.0, 6.0]);
        let y = sys.step(&x);
        assert_relative_eq!(y[0], 2.0);
        assert_relative_eq!(y[1], 3.0);
    }
}
