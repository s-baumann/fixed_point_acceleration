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

#[derive(Debug)]
pub struct FixedPointOptions {
    pub algorithm: Algorithm,
    pub threshold: f64,
    pub max_iter: usize,
    pub max_m: usize,
    pub extrapolation_period: usize,
    pub dampening: f64,
    pub print_reports: bool,
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
            print_reports: false,
        }
    }
}

#[derive(Debug)]
pub struct FixedPointResults {
    pub inputs: Vec<Array1<f64>>,
    pub outputs: Vec<Array1<f64>>,
    pub convergence_vector: Vec<f64>,
    pub status: String,
}