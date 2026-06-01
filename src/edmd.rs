//! Extended Dynamic Mode Decomposition (EDMD).
//!
//! EDMD lifts the state into a higher-dimensional feature space using a dictionary
//! of observables, then applies DMD in the lifted space.

use nalgebra::{DMatrix, DVector};
use crate::koopman::KoopmanOperator;
use crate::dmd::{pseudo_inverse, DmdResult};

/// A dictionary of observables for lifting states to feature space.
pub trait Dictionary: Clone {
    fn dim(&self) -> usize;
    fn eval(&self, state: &DVector<f64>) -> DVector<f64>;

    fn eval_batch(&self, states: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.dim();
        let m = states.ncols();
        let mut result = DMatrix::zeros(n, m);
        for i in 0..m {
            let col = states.column(i).clone_owned();
            let evals = self.eval(&col);
            for j in 0..n {
                result[(j, i)] = evals[j];
            }
        }
        result
    }
}

/// Polynomial dictionary: monomials up to a given degree.
#[derive(Debug, Clone)]
pub struct PolynomialDictionary {
    pub state_dim: usize,
    pub degree: usize,
    pub exponents: Vec<Vec<usize>>,
}

impl PolynomialDictionary {
    pub fn new(state_dim: usize, degree: usize) -> Self {
        let exponents = generate_monomials(state_dim, degree);
        Self { state_dim, degree, exponents }
    }
}

fn generate_monomials(dim: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0usize; dim];
    generate_recursive(dim, degree, 0, &mut current, &mut result);
    result
}

fn generate_recursive(dim: usize, degree: usize, idx: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if idx == dim {
        result.push(current.clone());
        return;
    }
    let used: usize = current.iter().sum();
    for d in 0..=(degree - used) {
        current[idx] = d;
        generate_recursive(dim, degree, idx + 1, current, result);
    }
    current[idx] = 0;
}

impl Dictionary for PolynomialDictionary {
    fn dim(&self) -> usize { self.exponents.len() }

    fn eval(&self, state: &DVector<f64>) -> DVector<f64> {
        let vals: Vec<f64> = self.exponents.iter().map(|exp| {
            let mut val = 1.0;
            for (i, &e) in exp.iter().enumerate() {
                val *= state[i].powi(e as i32);
            }
            val
        }).collect();
        DVector::from_vec(vals)
    }
}

/// Radial Basis Function (RBF) dictionary.
#[derive(Debug, Clone)]
pub struct RbfDictionary {
    pub centers: Vec<DVector<f64>>,
    pub epsilon: f64,
}

impl RbfDictionary {
    pub fn new(centers: Vec<DVector<f64>>, epsilon: f64) -> Self {
        Self { centers, epsilon }
    }

    pub fn from_data(data: &[DVector<f64>], num_centers: usize) -> Self {
        let centers: Vec<_> = data.iter().take(num_centers).cloned().collect();
        let epsilon = if centers.len() > 1 {
            let mut total_dist = 0.0;
            let mut count = 0;
            for i in 0..centers.len() {
                for j in (i + 1)..centers.len() {
                    total_dist += (&centers[i] - &centers[j]).norm();
                    count += 1;
                }
            }
            if count > 0 { total_dist / count as f64 } else { 1.0 }
        } else { 1.0 };
        Self { centers, epsilon: epsilon.max(1e-10) }
    }
}

impl Dictionary for RbfDictionary {
    fn dim(&self) -> usize { self.centers.len() }

    fn eval(&self, state: &DVector<f64>) -> DVector<f64> {
        let vals: Vec<f64> = self.centers.iter().map(|c| {
            let dist = (state - c).norm();
            (-self.epsilon * dist * dist).exp()
        }).collect();
        DVector::from_vec(vals)
    }
}

/// Extended DMD: apply DMD in the lifted space.
pub fn edmd<D: Dictionary>(x: &DMatrix<f64>, y: &DMatrix<f64>, dictionary: &D) -> Option<DmdResult> {
    let psi_x = dictionary.eval_batch(x);
    let psi_y = dictionary.eval_batch(y);

    let psi_x_pinv = pseudo_inverse(&psi_x)?;
    let k = &psi_y * psi_x_pinv;
    let koopman = KoopmanOperator::new(k);

    let eigenvalues = koopman.eigenvalues();
    let modes = psi_x;

    let reconstructed = &koopman.matrix * &modes;
    let error = (&psi_y - &reconstructed).norm();
    let y_norm = psi_y.norm();
    let relative_error = if y_norm > 1e-15 { error / y_norm } else { 0.0 };

    Some(DmdResult { koopman, eigenvalues, modes, relative_error })
}

/// Kernel EDMD using a kernel function.
pub fn kernel_edmd(
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
    kernel: impl Fn(&DVector<f64>, &DVector<f64>) -> f64,
) -> Option<DmdResult> {
    let m = x.ncols();
    if m == 0 { return None; }

    let mut g_xx = DMatrix::zeros(m, m);
    let mut g_yx = DMatrix::zeros(m, m);
    for i in 0..m {
        let xi = x.column(i).clone_owned();
        let yi = y.column(i).clone_owned();
        for j in 0..m {
            let xj = x.column(j).clone_owned();
            g_xx[(i, j)] = kernel(&xi, &xj);
            g_yx[(i, j)] = kernel(&yi, &xj);
        }
    }

    let g_xx_pinv = pseudo_inverse(&g_xx)?;
    let k_hat = g_yx * g_xx_pinv;
    let koopman = KoopmanOperator::new(k_hat);
    let eigenvalues = koopman.eigenvalues();

    Some(DmdResult { koopman, eigenvalues, modes: g_xx, relative_error: 0.0 })
}

/// Gaussian RBF kernel.
pub fn gaussian_kernel(x: &DVector<f64>, y: &DVector<f64>, sigma: f64) -> f64 {
    let dist_sq = (x - y).norm_squared();
    (-dist_sq / (2.0 * sigma * sigma)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_degree_1() {
        let dict = PolynomialDictionary::new(2, 1);
        assert_eq!(dict.dim(), 3);
    }

    #[test]
    fn test_polynomial_degree_2() {
        let dict = PolynomialDictionary::new(2, 2);
        assert_eq!(dict.dim(), 6);
    }

    #[test]
    fn test_polynomial_eval_constant() {
        let dict = PolynomialDictionary::new(2, 2);
        let state = DVector::from_vec(vec![3.0, 4.0]);
        let evals = dict.eval(&state);
        assert_relative_eq!(evals[0], 1.0);
    }

    #[test]
    fn test_polynomial_eval_linear() {
        let dict = PolynomialDictionary::new(2, 1);
        let state = DVector::from_vec(vec![3.0, 4.0]);
        let evals = dict.eval(&state);
        // Order depends on recursion: just check all expected values are present
        let vals: Vec<f64> = evals.iter().copied().collect();
        assert!(vals.contains(&1.0));
        assert!(vals.contains(&3.0));
        assert!(vals.contains(&4.0));
    }

    #[test]
    fn test_rbf_eval() {
        let centers = vec![DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![1.0, 0.0])];
        let dict = RbfDictionary::new(centers, 1.0);
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let evals = dict.eval(&state);
        assert_relative_eq!(evals[0], 1.0);
        assert!(evals[1] < 1.0);
    }

    #[test]
    fn test_rbf_from_data() {
        let data = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let dict = RbfDictionary::from_data(&data, 3);
        assert_eq!(dict.dim(), 3);
    }

    #[test]
    fn test_edmd_linear() {
        let dict = PolynomialDictionary::new(1, 1);
        let x = DMatrix::from_row_slice(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let y = DMatrix::from_row_slice(1, 4, &[2.0, 4.0, 6.0, 8.0]);
        let result = edmd(&x, &y, &dict).unwrap();
        assert!(result.relative_error < 1e-6);
    }

    #[test]
    fn test_edmd_2d() {
        let dict = PolynomialDictionary::new(2, 2);
        let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
        let y = DMatrix::from_row_slice(2, 4, &[0.5, 1.0, 1.5, 2.0, 0.0, 0.5, 1.0, 1.5]);
        let result = edmd(&x, &y, &dict).unwrap();
        assert!(result.relative_error < 1e-4);
    }

    #[test]
    fn test_gaussian_kernel_same() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        assert_relative_eq!(gaussian_kernel(&x, &x, 1.0), 1.0);
    }

    #[test]
    fn test_gaussian_kernel_different() {
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let y = DVector::from_vec(vec![1.0, 0.0]);
        let k = gaussian_kernel(&x, &y, 1.0);
        assert!(k < 1.0 && k > 0.0);
    }

    #[test]
    fn test_kernel_edmd() {
        let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
        let y = x.scale(0.5);
        let result = kernel_edmd(&x, &y, |a, b| gaussian_kernel(a, b, 2.0));
        assert!(result.is_some());
    }

    #[test]
    fn test_polynomial_degree_1_dim3() {
        let dict = PolynomialDictionary::new(3, 1);
        assert_eq!(dict.dim(), 4);
    }

    #[test]
    fn test_generate_monomials() {
        let monoms = generate_monomials(2, 2);
        assert_eq!(monoms.len(), 6);
    }
}
