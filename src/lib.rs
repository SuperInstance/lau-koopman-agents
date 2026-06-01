//! # lau-koopman-agents
//!
//! Koopman operator theory for dynamical systems and agent learning.
//!
//! The Koopman operator is an infinite-dimensional linear operator that governs
//! the evolution of observable functions of a nonlinear dynamical system.
//! This crate provides tools for:
//! - Koopman operator approximation via Dynamic Mode Decomposition (DMD)
//! - Extended DMD (EDMD) with dictionary learning
//! - Koopman eigenfunctions, modes, and invariant subspaces
//! - Connection to spectral theory

pub mod koopman;
pub mod dmd;
pub mod edmd;
pub mod eigenfunctions;
pub mod invariant_subspaces;
pub mod spectral;
pub mod agent_learning;

pub use koopman::*;
pub use dmd::*;
pub use edmd::*;
pub use eigenfunctions::*;
pub use invariant_subspaces::*;
pub use spectral::*;
pub use agent_learning::*;
