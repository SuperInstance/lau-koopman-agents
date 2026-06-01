//! Koopman eigenfunctions and modes.
//!
//! Koopman eigenfunctions φ satisfy K φ = λ φ.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use crate::koopman::KoopmanOperator;

/// A Koopman eigenfunction with its eigenvalue.
#[derive(Debug, Clone)]
pub struct KoopmanEigenfunction {
    pub eigenvalue: Complex64,
    pub values: DVector<f64>,
}

impl KoopmanEigenfunction {
    pub fn new(eigenvalue: Complex64, values: DVector<f64>) -> Self {
        Self { eigenvalue, values }
    }

    pub fn growth_rate(&self) -> f64 { self.eigenvalue.norm() }
    pub fn frequency(&self) -> f64 { self.eigenvalue.arg() }
    pub fn is_growing(&self) -> bool { self.eigenvalue.norm() > 1.0 + 1e-10 }
    pub fn is_decaying(&self) -> bool { self.eigenvalue.norm() < 1.0 - 1e-10 }
    pub fn is_marginal(&self) -> bool { (self.eigenvalue.norm() - 1.0).abs() < 1e-10 }

    /// Evaluate the eigenfunction at future time: φ(x_k) = λ^k φ(x₀).
    pub fn evolve(&self, k: i32) -> DVector<Complex64> {
        let lambda_k = self.eigenvalue.powi(k);
        let n = self.values.nrows();
        let mut result = DVector::zeros_complex(n);
        for i in 0..n {
            result[i] = lambda_k * self.values[i];
        }
        result
    }
}

/// Trait to create complex vectors easily.
trait DVectorComplex {
    fn zeros_complex(n: usize) -> DVector<Complex64>;
}

impl DVectorComplex for DVector<Complex64> {
    fn zeros_complex(n: usize) -> DVector<Complex64> {
        DVector::from_element(n, Complex64::new(0.0, 0.0))
    }
}

/// Compute Koopman eigenfunctions from a Koopman operator approximation.
pub fn compute_eigenfunctions(koopman: &KoopmanOperator) -> Vec<KoopmanEigenfunction> {
    let eigenvalues = koopman.eigenvalues();
    eigenvalues.into_iter().map(|λ| {
        let v = approximate_eigenvector(&koopman.matrix, &λ, 50);
        KoopmanEigenfunction::new(λ, v)
    }).collect()
}

fn approximate_eigenvector(matrix: &DMatrix<f64>, eigenvalue: &Complex64, iterations: usize) -> DVector<f64> {
    let n = matrix.nrows();
    if n == 0 { return DVector::zeros(0); }

    let shift = eigenvalue.re;
    let shifted = matrix.clone() - DMatrix::from_diagonal_element(n, n, shift);
    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());

    for _ in 0..iterations {
        let mut w = v.clone();
        for _ in 0..5 {
            let residual = &v - &shifted * &w;
            w += residual.scale(0.1);
        }
        let norm = w.norm();
        if norm > 1e-15 { v = w / norm; }
    }
    v
}

/// Koopman mode decomposition.
#[derive(Debug, Clone)]
pub struct KoopmanModeDecomposition {
    pub eigenvalues: Vec<Complex64>,
    pub modes: Vec<DVector<f64>>,
    pub amplitudes: Vec<f64>,
}

impl KoopmanModeDecomposition {
    /// Reconstruct the state at time step k.
    pub fn reconstruct(&self, k: usize) -> DVector<f64> {
        if self.modes.is_empty() { return DVector::zeros(0); }
        let n = self.modes[0].nrows();
        let mut result = DVector::zeros(n);
        for i in 0..self.eigenvalues.len() {
            let lambda_k = self.eigenvalues[i].powi(k as i32);
            let coeff = self.amplitudes[i] * lambda_k.re;
            result += self.modes[i].scale(coeff);
        }
        result
    }

    pub fn num_modes(&self) -> usize { self.eigenvalues.len() }

    pub fn sorted_by_amplitude(&mut self) {
        let mut indexed: Vec<_> = (0..self.eigenvalues.len()).collect();
        indexed.sort_by(|&a, &b| {
            self.amplitudes[b].partial_cmp(&self.amplitudes[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let eigs: Vec<_> = indexed.iter().map(|&i| self.eigenvalues[i]).collect();
        let modes: Vec<_> = indexed.iter().map(|&i| self.modes[i].clone()).collect();
        let amps: Vec<_> = indexed.iter().map(|&i| self.amplitudes[i]).collect();
        self.eigenvalues = eigs;
        self.modes = modes;
        self.amplitudes = amps;
    }

    pub fn truncate(&mut self, r: usize) {
        self.sorted_by_amplitude();
        self.eigenvalues.truncate(r);
        self.modes.truncate(r);
        self.amplitudes.truncate(r);
    }
}

/// Compute mode decomposition from snapshot data.
pub fn mode_decomposition(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Option<KoopmanModeDecomposition> {
    use crate::dmd::dmd;
    let dmd_result = dmd(x, y)?;
    let eigenvalues = dmd_result.koopman.eigenvalues();
    let n = dmd_result.koopman.num_observables;

    let mut modes = Vec::new();
    let mut amplitudes = Vec::new();
    for λ in &eigenvalues {
        let v = approximate_eigenvector(&dmd_result.koopman.matrix, λ, 50);
        modes.push(v.clone());
        if x.nrows() == n && x.ncols() > 0 {
            let x0 = x.column(0).clone_owned();
            amplitudes.push(v.dot(&x0));
        } else {
            amplitudes.push(1.0);
        }
    }

    Some(KoopmanModeDecomposition { eigenvalues, modes, amplitudes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_growth_rate() {
        let ef = KoopmanEigenfunction::new(Complex64::new(2.0, 0.0), DVector::from_vec(vec![1.0]));
        assert_relative_eq!(ef.growth_rate(), 2.0);
    }

    #[test]
    fn test_growing() {
        let ef = KoopmanEigenfunction::new(Complex64::new(2.0, 0.0), DVector::from_vec(vec![1.0]));
        assert!(ef.is_growing());
        assert!(!ef.is_decaying());
    }

    #[test]
    fn test_decaying() {
        let ef = KoopmanEigenfunction::new(Complex64::new(0.5, 0.0), DVector::from_vec(vec![1.0]));
        assert!(ef.is_decaying());
        assert!(!ef.is_growing());
    }

    #[test]
    fn test_marginal() {
        let ef = KoopmanEigenfunction::new(Complex64::from_polar(1.0, 0.5), DVector::from_vec(vec![1.0]));
        assert!(ef.is_marginal());
    }

    #[test]
    fn test_frequency() {
        let ef = KoopmanEigenfunction::new(
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
            DVector::from_vec(vec![1.0]),
        );
        assert_relative_eq!(ef.frequency(), std::f64::consts::FRAC_PI_4, epsilon = 1e-10);
    }

    #[test]
    fn test_evolve() {
        let ef = KoopmanEigenfunction::new(Complex64::new(2.0, 0.0), DVector::from_vec(vec![1.0]));
        let evolved = ef.evolve(3);
        assert_relative_eq!(evolved[0].re, 8.0);
    }

    #[test]
    fn test_compute_eigenfunctions() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 0.5));
        let efs = compute_eigenfunctions(&k);
        assert_eq!(efs.len(), 2);
    }

    #[test]
    fn test_mode_decomposition() {
        let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
        let y = x.scale(0.5);
        let md = mode_decomposition(&x, &y).unwrap();
        assert!(md.num_modes() > 0);
    }

    #[test]
    fn test_reconstruction() {
        let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
        let y = x.scale(0.5);
        let md = mode_decomposition(&x, &y).unwrap();
        let recon = md.reconstruct(0);
        assert!(recon.norm() > 0.0);
    }

    #[test]
    fn test_sort_truncate() {
        let mut md = KoopmanModeDecomposition {
            eigenvalues: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            modes: vec![DVector::from_vec(vec![1.0]), DVector::from_vec(vec![2.0])],
            amplitudes: vec![0.1, 0.9],
        };
        md.sorted_by_amplitude();
        assert_relative_eq!(md.amplitudes[0], 0.9);
        md.truncate(1);
        assert_eq!(md.num_modes(), 1);
    }

    #[test]
    fn test_complex_evolve() {
        let ef = KoopmanEigenfunction::new(
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_2),
            DVector::from_vec(vec![1.0]),
        );
        let evolved = ef.evolve(4); // i^4 = 1
        assert_relative_eq!(evolved[0].re, 1.0, epsilon = 1e-10);
    }
}
