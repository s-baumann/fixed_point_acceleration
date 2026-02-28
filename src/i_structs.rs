use ndarray::Array1;

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    Simple, // Picard Iteration
    Newton, // Newton's Method applied to g(x) = f(x) - x
    Aitken, // Aitken's Δ² method
    Anderson, // Anderson Acceleration
    MPE, // Minimal Polynomial Extrapolation
    RRE, // Reduced Rank Extrapolation
    VEA, // Vector Epsilon Algorithm
    SEA, // Scalar Epsilon Algorithm
}

/// Why the iteration loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationStatus {
    /// The mean absolute residual `|g(x) - x|` fell below `threshold`.
    Converged,
    /// The loop ran for `max_iter` iterations without converging.
    MaxIterationsReached,
}

#[derive(Debug)]
pub struct FixedPointOptions {
    pub algorithm: Algorithm,
    pub threshold: f64,
    pub max_iter: usize,
    pub max_m: usize,
    pub extrapolation_period: usize,
    pub dampening: f64,
}

impl Default for FixedPointOptions {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Anderson,
            threshold: 1e-10,
            max_iter: 1000,
            max_m: 10,
            extrapolation_period: 7,
            dampening: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct FixedPointResults {
    pub inputs: Vec<Array1<f64>>,
    pub outputs: Vec<Array1<f64>>,
    pub convergence_vector: Vec<f64>,
    pub status: TerminationStatus,
}