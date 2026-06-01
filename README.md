# lau-koopman-agents

**Koopman operator theory for dynamical systems and agent learning in Rust.**

This crate provides finite-dimensional approximations of the Koopman operator — the infinite-dimensional linear operator that governs the evolution of observable functions over a nonlinear dynamical system. It implements Dynamic Mode Decomposition (DMD), Extended DMD (EDMD) with dictionary learning, eigenfunction analysis, invariant subspace detection, spectral theory, and agent-based learning — all backed by 82 tests.

---

## What This Does

Most nonlinear dynamical systems are hard to analyze directly. The Koopman operator sidesteps this by *lifting* the problem: instead of tracking states through a nonlinear map, you track *observable functions of those states* through a **linear** operator. The catch is that this operator lives in an infinite-dimensional space — so in practice you approximate it.

This crate gives you the tools to:

- **Learn a linear model from trajectory data** (DMD / EDMD)
- **Forecast future states** using the learned operator
- **Extract eigenvalues, eigenfunctions, and Koopman modes** for spectral analysis
- **Detect invariant subspaces** and verify their structure
- **Analyze stability, ergodicity, chaos, and mixing** from operator spectra
- **Train agents online** with streaming data via gradient updates

---

## Key Idea

For a dynamical system **x_{k+1} = F(x_k)**, the Koopman operator **K** acts on observables **g** as:

```
(K g)(x) = g(F(x))
```

K is always linear — even when F is not. Its eigenvalues determine stability, its modes determine the coherent structures in the flow, and its invariant subspaces tell you when a finite approximation is exact.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-koopman-agents = "0.1.0"
```

Or use it as a local dependency:

```toml
[dependencies]
lau-koopman-agents = { path = "../lau-koopman-agents" }
```

Requires **Rust 2021 edition**. Dependencies: `nalgebra` (with serde), `num-complex`, `serde`.

---

## Quick Start

### Learn a Koopman model from trajectory data

```rust
use nalgebra::{DMatrix, DVector};
use lau_koopman_agents::*;

// Define a linear system x_{k+1} = 0.5 * x_k
let system = LinearSystem::new(DMatrix::from_row_slice(1, 1, &[0.5]));
let x0 = DVector::from_vec(vec![1.0]);

// Generate a trajectory
let trajectory = generate_trajectory(&system, &x0, 20);

// Learn a Koopman model
let model = KoopmanAgentModel::from_trajectory(&trajectory).unwrap();

// Predict future states
let predictions = model.predict_multistep(&DVector::from_vec(vec![4.0]), 5);
println!("Next state: {:?}", predictions[0]); // ≈ 2.0
```

### Dynamic Mode Decomposition (DMD)

```rust
use lau_koopman_agents::dmd::*;

// Snapshot matrices: X = current states, Y = next states
let x = DMatrix::from_row_slice(1, 4, &[1.0, 2.0, 3.0, 4.0]);
let y = DMatrix::from_row_slice(1, 4, &[2.0, 4.0, 6.0, 8.0]);

let result = dmd(&x, &y).unwrap();
println!("Eigenvalues: {:?}", result.eigenvalues);
println!("Relative error: {}", result.relative_error); // ≈ 0

// Forecast ahead
let forecast = dmd_forecast(&result, &DVector::from_vec(vec![5.0]), 3);
```

### Extended DMD with polynomial lifting

```rust
use lau_koopman_agents::edmd::*;

let dict = PolynomialDictionary::new(2, 2); // 2D state, degree-2 monomials
let x = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0]);
let y = x.scale(0.5);

let result = edmd(&x, &y, &dict).unwrap();
```

### Spectral analysis

```rust
use lau_koopman_agents::spectral::*;

let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.5]));
let props = SpectralProperties::from_koopman(&k);
println!("Spectral gap: {}", props.spectral_gap);    // 0.5
println!("Mixing time: {}", props.mixing_time);       // 2.0
println!("Stable? {}", props.is_stable);              // true

println!("Chaotic? {}", is_chaotic(&k));              // false
println!("Lyapunov exponent: {}", lyapunov_exponent_estimate(&k));
```

### Online learning

```rust
use lau_koopman_agents::agent_learning::*;

let mut learner = OnlineKoopmanLearner::new(0.01);

for i in 0..200 {
    let x = DVector::from_vec(vec![(i as f64 * 0.1).sin()]);
    let y = DVector::from_vec(vec![0.9 * x[0]]);
    learner.update(&x, &y);
}

let pred = learner.predict(&DVector::from_vec(vec![1.0])).unwrap();
```

---

## API Reference

### `koopman` — Core Operator Theory

| Type / Function | Description |
|---|---|
| `DynamicalSystem` (trait) | Interface for dynamical systems: `step(&state) → next_state` |
| `LinearSystem` | x_{k+1} = A x_k with a matrix `A` |
| `FnSystem` | Nonlinear system from a closure |
| `KoopmanOperator` | Finite-dimensional Koopman matrix with eigenvalues, stability check, spectral radius, modes |
| `generate_trajectory` | Run a system for N steps from an initial state |
| `stack_snapshots` | Convert a trajectory into (X, Y) snapshot matrices |

### `dmd` — Dynamic Mode Decomposition

| Type / Function | Description |
|---|---|
| `dmd(x, y)` | Standard DMD: solves min ‖Y − AX‖_F via A = Y X⁺ |
| `dmd_from_trajectory` | DMD directly from trajectory data |
| `dmd_forecast` | Multi-step prediction using a DMD result |
| `subspace_dmd` | Rank-reduced DMD via SVD projection |
| `pseudo_inverse` | Moore-Penrose pseudoinverse via SVD |
| `DmdResult` | Holds the Koopman operator, eigenvalues, modes, and reconstruction error |

### `edmd` — Extended DMD

| Type / Function | Description |
|---|---|
| `Dictionary` (trait) | Observable dictionary: `eval(state) → lifted_features` |
| `PolynomialDictionary` | Monomials up to a given degree |
| `RbfDictionary` | Radial basis functions centered on data points |
| `edmd(x, y, dict)` | DMD in the lifted feature space |
| `kernel_edmd` | Kernelized EDMD using a custom kernel function |
| `gaussian_kernel` | Gaussian RBF kernel utility |

### `eigenfunctions` — Eigenfunctions & Modes

| Type / Function | Description |
|---|---|
| `KoopmanEigenfunction` | Eigenpair (λ, φ) with growth rate, frequency, evolve |
| `compute_eigenfunctions` | Extract all eigenfunctions from a Koopman operator |
| `KoopmanModeDecomposition` | Mode decomposition with reconstruction, sorting, truncation |
| `mode_decomposition` | Compute modes from snapshot data |

### `invariant_subspaces` — Invariant Subspace Detection

| Type / Function | Description |
|---|---|
| `InvariantSubspace` | A subspace V with K V ⊆ V, with projection and invariance verification |
| `find_invariant_subspace` | Find an approximately invariant subspace via SVD |
| `refine_invariant_subspace` | Power-iteration refinement of an invariant subspace |
| `check_invariance` | Test whether a set of observables forms an invariant subspace |

### `spectral` — Spectral Theory

| Type / Function | Description |
|---|---|
| `SpectralProperties` | Spectral radius, stability, spectral gap, mixing time |
| `ModeSpectrum` | Mode classification (growing / decaying / marginal) |
| `ergodicity_check` | Test ergodicity via unit-circle eigenvalue count |
| `lyapunov_exponent_estimate` | Estimate Lyapunov exponent from dominant eigenvalue |
| `is_chaotic` | Check for chaos (positive Lyapunov exponent) |
| `spectral_decomposition` | Cluster eigenvalues by magnitude |

### `agent_learning` — Agent-Based Learning

| Type / Function | Description |
|---|---|
| `KoopmanAgentModel` | Learned model with predict, multistep prediction, test error, eigenfunctions |
| `OnlineKoopmanLearner` | Streaming gradient-descent learner |
| `model_selection` | Compare DMD and EDMD models at various polynomial degrees |

---

## How It Works

1. **Data collection**: Generate or observe a trajectory `{x_0, x_1, ..., x_T}` from a dynamical system.
2. **Snapshot matrices**: Stack into `X = [x_0, ..., x_{T-1}]` and `Y = [x_1, ..., x_T]`.
3. **Operator recovery**: Solve `A = Y X⁺` (DMD) or lift to feature space first (EDMD).
4. **Analysis**: Eigenvalues → stability, modes → coherent structures, spectral gap → mixing rate.
5. **Prediction**: Apply `A` iteratively: `x_{k+1} = A x_k`.

The Koopman operator is the "God's-eye view" of a dynamical system: it linearizes everything, at the cost of infinite dimensionality. This crate finds finite approximations that are good enough to be useful.

---

## The Math

### The Koopman Operator

Given a dynamical system x_{k+1} = F(x_k) on a state space M, the **Koopman operator** K : L²(M) → L²(M) is defined by:

```
(Kg)(x) = g(F(x))   for all observables g
```

K is always **linear** in the space of observables, even when F is nonlinear. Its spectral decomposition reveals the intrinsic time scales and coherent structures of the system.

### Dynamic Mode Decomposition (DMD)

Given snapshot matrices X, Y, DMD solves:

```
A = argmin_B ‖Y - BX‖_F = Y X⁺
```

where X⁺ is the Moore-Penrose pseudoinverse. The eigenvalues of A approximate the Koopman eigenvalues, and the eigenvectors (projected back to physical space) are the **Koopman modes**.

### Extended DMD (EDMD)

EDMD lifts the state into a **dictionary of observables** {ψ₁, ..., ψ_N} before applying DMD:

```
Ψ(X) = [ψ(x₁), ..., ψ(x_m)]     (lifted snapshot matrix)
K = Ψ(Y) Ψ(X)⁺                   (Koopman in feature space)
```

With a rich enough dictionary, this captures nonlinear dynamics that standard DMD misses.

### Eigenfunctions & Modes

A **Koopman eigenfunction** φ satisfies:

```
Kφ = λφ    ⟹    φ(x_k) = λ^k φ(x₀)
```

The system state decomposes as:

```
x_k = Σ_i  λ_i^k  φ_i(x₀)  v_i
```

where v_i are the **Koopman modes** — spatial patterns oscillating at the eigenfrequencies.

### Spectral Theory

- **Spectral radius** ρ(K) = max |λ_i| determines stability
- **Spectral gap** = 1 − |λ₂| controls mixing rate (larger gap → faster convergence to equilibrium)
- **Lyapunov exponent** ≈ ln(ρ(K)): positive implies chaos
- **Ergodicity**: a single eigenvalue at λ = 1 with multiplicity 1

### Invariant Subspaces

A subspace V of observables is **Koopman-invariant** if K(V) ⊆ V. These subspaces are the "natural" finite-dimensional truncations — they preserve the linear structure exactly. Finding them is equivalent to finding observables whose span is closed under the dynamics.

---

## Test Coverage

**82 tests** across all modules:

| Module | Tests | What's covered |
|---|---|---|
| `koopman` | 14 | Linear/fn systems, trajectory generation, snapshot stacking, eigenvalues, stability, modes |
| `dmd` | 10 | Standard DMD, subspace DMD, pseudoinverse, forecasting, edge cases |
| `edmd` | 14 | Polynomial & RBF dictionaries, EDMD, kernel EDMD, Gaussian kernel |
| `eigenfunctions` | 12 | Growth/decay/marginal classification, frequency, evolution, mode decomposition |
| `invariant_subspaces` | 11 | Projection, containment, invariance verification, energy fraction, refinement |
| `spectral` | 14 | Spectral properties, mode spectrum, ergodicity, Lyapunov exponents, chaos |
| `agent_learning` | 11 | Model learning, prediction, online updates, model selection, edge cases |

Run them with:

```bash
cargo test
```

---

## License

MIT
