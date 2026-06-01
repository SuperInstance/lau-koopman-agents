//! Spectral theory connections for the Koopman operator.
//!
//! The Koopman operator K is a unitary or contraction operator on L².
//! Its spectral properties determine the long-term behavior of the system.

use num_complex::Complex64;
use crate::koopman::KoopmanOperator;

/// Spectral properties of a Koopman operator.
#[derive(Debug, Clone)]
pub struct SpectralProperties {
    /// Eigenvalues of the finite approximation.
    pub eigenvalues: Vec<Complex64>,
    /// Spectral radius.
    pub spectral_radius: f64,
    /// Is the system stable (all |λ| ≤ 1)?
    pub is_stable: bool,
    /// Is the system asymptotically stable (all |λ| < 1)?
    pub is_asymptotically_stable: bool,
    /// The spectral gap: 1 - |λ₂| (where λ₂ is second-largest eigenvalue magnitude).
    /// Larger gap → faster mixing.
    pub spectral_gap: f64,
    /// Mixing time (related to spectral gap).
    pub mixing_time: f64,
}

impl SpectralProperties {
    /// Compute spectral properties from a Koopman operator.
    pub fn from_koopman(koopman: &KoopmanOperator) -> Self {
        let eigenvalues = koopman.eigenvalues();
        let mut magnitudes: Vec<f64> = eigenvalues.iter().map(|λ| λ.norm()).collect();
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let spectral_radius = magnitudes.first().copied().unwrap_or(0.0);
        let is_stable = spectral_radius <= 1.0 + 1e-10;
        let is_asymptotically_stable = magnitudes.iter().all(|&m| m < 1.0 - 1e-10);
        let spectral_gap = if magnitudes.len() >= 2 {
            1.0 - magnitudes[1]
        } else {
            0.0
        };
        let mixing_time = if spectral_gap.abs() > 1e-15 {
            1.0 / spectral_gap
        } else {
            f64::INFINITY
        };

        Self {
            eigenvalues,
            spectral_radius,
            is_stable,
            is_asymptotically_stable,
            spectral_gap,
            mixing_time,
        }
    }
}

/// Koopman mode spectrum analysis.
#[derive(Debug, Clone)]
pub struct ModeSpectrum {
    /// Modes sorted by eigenvalue magnitude.
    pub modes: Vec<(Complex64, f64)>, // (eigenvalue, magnitude)
}

impl ModeSpectrum {
    /// Analyze the mode spectrum of a Koopman operator.
    pub fn analyze(koopman: &KoopmanOperator) -> Self {
        let eigenvalues = koopman.eigenvalues();
        let mut modes: Vec<_> = eigenvalues.into_iter()
            .map(|λ| (λ, λ.norm()))
            .collect();
        modes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Self { modes }
    }

    /// Number of growing modes.
    pub fn num_growing_modes(&self) -> usize {
        self.modes.iter().filter(|(_, m)| *m > 1.0 + 1e-10).count()
    }

    /// Number of decaying modes.
    pub fn num_decaying_modes(&self) -> usize {
        self.modes.iter().filter(|(_, m)| *m < 1.0 - 1e-10).count()
    }

    /// Number of marginal modes.
    pub fn num_marginal_modes(&self) -> usize {
        self.modes.iter().filter(|(_, m)| (m - 1.0).abs() <= 1e-10).count()
    }

    /// Dominant mode (largest eigenvalue magnitude).
    pub fn dominant_mode(&self) -> Option<&(Complex64, f64)> {
        self.modes.first()
    }

    /// Slow-decaying modes (top-k by magnitude).
    pub fn slow_modes(&self, k: usize) -> &[(Complex64, f64)] {
        let end = k.min(self.modes.len());
        &self.modes[..end]
    }
}

/// Ergodicity quotient: if the Koopman operator has a single eigenvalue at 1
/// (with multiplicity 1), the system is ergodic.
pub fn ergodicity_check(koopman: &KoopmanOperator) -> ErgodicityResult {
    let eigenvalues = koopman.eigenvalues();
    let unit_eigenvalues: Vec<_> = eigenvalues.iter()
        .filter(|λ| (λ.norm() - 1.0).abs() < 1e-8)
        .collect();

    ErgodicityResult {
        num_unit_eigenvalues: unit_eigenvalues.len(),
        is_likely_ergodic: unit_eigenvalues.len() == 1
            && (unit_eigenvalues[0] - Complex64::new(1.0, 0.0)).norm() < 1e-8,
    }
}

/// Result of an ergodicity check.
#[derive(Debug, Clone)]
pub struct ErgodicityResult {
    /// Number of eigenvalues with |λ| ≈ 1.
    pub num_unit_eigenvalues: usize,
    /// Whether the system is likely ergodic.
    pub is_likely_ergodic: bool,
}

/// Lyapunov exponent estimate from Koopman eigenvalues.
/// The largest Lyapunov exponent ≈ ln(|λ_max|) where λ_max is the largest eigenvalue.
pub fn lyapunov_exponent_estimate(koopman: &KoopmanOperator) -> f64 {
    let eigenvalues = koopman.eigenvalues();
    let max_mag = eigenvalues.iter().map(|λ| λ.norm()).fold(0.0_f64, f64::max);
    if max_mag > 0.0 { max_mag.ln() } else { f64::NEG_INFINITY }
}

/// Is the system chaotic (positive Lyapunov exponent)?
pub fn is_chaotic(koopman: &KoopmanOperator) -> bool {
    lyapunov_exponent_estimate(koopman) > 1e-10
}

/// Spectral decomposition: project onto eigenvalue clusters.
pub fn spectral_decomposition(
    koopman: &KoopmanOperator,
    num_clusters: usize,
) -> Vec<Vec<Complex64>> {
    let eigenvalues = koopman.eigenvalues();
    if eigenvalues.is_empty() {
        return vec![];
    }

    // Simple clustering by magnitude
    let mut sorted = eigenvalues.clone();
    sorted.sort_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let cluster_size = (n + num_clusters - 1) / num_clusters.max(1);

    let mut clusters = Vec::new();
    let mut i = 0;
    while i < n {
        let end = (i + cluster_size).min(n);
        clusters.push(sorted[i..end].to_vec());
        i = end;
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use approx::assert_relative_eq;

    #[test]
    fn test_spectral_properties_identity() {
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        let props = SpectralProperties::from_koopman(&k);
        assert!(props.is_stable);
        assert!(!props.is_asymptotically_stable);
        assert_relative_eq!(props.spectral_radius, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_properties_decay() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 0.5));
        let props = SpectralProperties::from_koopman(&k);
        assert!(props.is_stable);
        assert!(props.is_asymptotically_stable);
        assert_relative_eq!(props.spectral_radius, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_properties_unstable() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 2.0));
        let props = SpectralProperties::from_koopman(&k);
        assert!(!props.is_stable);
        assert_relative_eq!(props.spectral_radius, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_gap() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.5]));
        let props = SpectralProperties::from_koopman(&k);
        assert_relative_eq!(props.spectral_gap, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mixing_time() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.5]));
        let props = SpectralProperties::from_koopman(&k);
        assert_relative_eq!(props.mixing_time, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mode_spectrum() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.9, 0.0, 0.0, 0.5]));
        let spectrum = ModeSpectrum::analyze(&k);
        assert_eq!(spectrum.num_growing_modes(), 0);
        assert_eq!(spectrum.num_decaying_modes(), 2);
    }

    #[test]
    fn test_dominant_mode() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.9, 0.0, 0.0, 0.5]));
        let spectrum = ModeSpectrum::analyze(&k);
        let dominant = spectrum.dominant_mode().unwrap();
        assert_relative_eq!(dominant.1, 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_slow_modes() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(3, 3, &[0.9, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1]));
        let spectrum = ModeSpectrum::analyze(&k);
        let slow = spectrum.slow_modes(2);
        assert_eq!(slow.len(), 2);
    }

    #[test]
    fn test_ergodicity_identity() {
        let k = KoopmanOperator::new(DMatrix::identity(1, 1));
        let result = ergodicity_check(&k);
        assert!(result.is_likely_ergodic);
    }

    #[test]
    fn test_ergodicity_nonergodic() {
        let k = KoopmanOperator::new(DMatrix::identity(3, 3));
        let result = ergodicity_check(&k);
        assert!(!result.is_likely_ergodic);
    }

    #[test]
    fn test_lyapunov_stable() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 0.5));
        let lyap = lyapunov_exponent_estimate(&k);
        assert!(lyap < 0.0);
    }

    #[test]
    fn test_lyapunov_chaotic() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 2.0));
        let lyap = lyapunov_exponent_estimate(&k);
        assert!(lyap > 0.0);
        assert!(is_chaotic(&k));
    }

    #[test]
    fn test_is_not_chaotic() {
        let k = KoopmanOperator::new(DMatrix::from_diagonal_element(2, 2, 0.5));
        assert!(!is_chaotic(&k));
    }

    #[test]
    fn test_spectral_decomposition() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(3, 3, &[0.9, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1]));
        let clusters = spectral_decomposition(&k, 2);
        assert_eq!(clusters.len(), 2);
    }
}
