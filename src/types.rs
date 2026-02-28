use ndarray::Array1;

/// Acceleration algorithm to use in the fixed-point iteration.
#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    /// Successive substitution (Picard iteration). No acceleration; next input = `g(x)`.
    Simple,
    /// Secant-based acceleration applied element-wise every 3rd iteration.
    Newton,
    /// Aitken's Δ² extrapolation applied element-wise every 3rd iteration.
    Aitken,
    /// Anderson mixing over a sliding window of the `max_m` most recent iterates.
    Anderson,
    /// Minimal Polynomial Extrapolation, applied every `extrapolation_period` iterations.
    MPE,
    /// Reduced Rank Extrapolation, applied every `extrapolation_period` iterations.
    RRE,
    /// Vector Epsilon Algorithm (Wynn ε, pseudoinverse), applied every `extrapolation_period` iterations.
    VEA,
    /// Scalar Epsilon Algorithm (Wynn ε, element-wise), applied every `extrapolation_period` iterations.
    SEA,
}

/// Why the fixed-point iteration loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationStatus {
    /// The mean absolute residual `mean(|g(x) − x|)` fell below `threshold`.
    Converged,
    /// The loop ran for `max_iter` iterations without converging.
    MaxIterationsReached,
    /// The acceleration step produced a non-finite value (NaN or ±Inf), typically
    /// due to a near-zero denominator in Aitken or Newton acceleration.
    /// The returned [`FixedPointResults`] contains all iterates up to the failure point
    /// and can be passed to [`fixed_point_from`](crate::fixed_point_from) to resume
    /// with a different algorithm.
    NumericalFailure,
    /// The function `g` returned a non-finite value (NaN or ±Inf) for the current input,
    /// for example when evaluating `sqrt` at a negative number.
    /// The returned [`FixedPointResults`] contains all valid iterates up to (but not
    /// including) the failing call.
    FunctionFailure,
}

impl std::fmt::Display for TerminationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Converged            => write!(f, "Converged"),
            Self::MaxIterationsReached => write!(f, "MaxIterationsReached"),
            Self::NumericalFailure     => write!(f, "NumericalFailure"),
            Self::FunctionFailure      => write!(f, "FunctionFailure"),
        }
    }
}

/// Configuration for [`fixed_point`](crate::fixed_point) and [`fixed_point_from`](crate::fixed_point_from).
#[derive(Debug, Clone)]
pub struct FixedPointOptions {
    /// Acceleration algorithm to use. Default: `Anderson`.
    pub algorithm: Algorithm,
    /// Convergence threshold: iteration stops when `mean(|g(x) − x|)` drops below this value.
    /// Default: `1e-10`.
    pub threshold: f64,
    /// Hard cap on the number of iterations. Default: `1000`.
    pub max_iter: usize,
    /// History window for Anderson, MPE, RRE, SEA, and VEA. Default: `10`.
    pub max_m: usize,
    /// How often MPE, RRE, SEA, and VEA fire (in iterations); simple substitution fills
    /// the gaps. Default: `7`.
    pub extrapolation_period: usize,
    /// Step-size damping factor in `(0, 1]`. `1.0` means no damping; the proposed
    /// iterate is blended as `(1 − d)·x_old + d·proposed`. Default: `1.0`.
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

/// Results returned by [`fixed_point`](crate::fixed_point) and [`fixed_point_from`](crate::fixed_point_from).
#[derive(Debug, Clone)]
pub struct FixedPointResults {
    /// Input vector fed to `g` at each iteration.
    pub inputs: Vec<Array1<f64>>,
    /// Output `g(x)` at each iteration.
    pub outputs: Vec<Array1<f64>>,
    /// Mean absolute residual `mean(|g(x) − x|)` at each iteration.
    pub convergence_vector: Vec<f64>,
    /// Reason the iteration loop stopped.
    pub status: TerminationStatus,
}
