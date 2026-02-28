//! Fixed-point iteration with acceleration algorithms.
//!
//! Given a function `g: ℝⁿ → ℝⁿ`, finds `x*` such that `g(x*) = x*`.
//! Acceleration methods dramatically reduce the number of iterations required
//! compared to plain successive substitution.
//!
//! # Quick start
//!
//! ```
//! use fixed_point_acceleration::{fixed_point, Algorithm, FixedPointOptions, TerminationStatus};
//! use ndarray::array;
//!
//! // Find the Dottie number: the fixed point of cos(x) ≈ 0.7391
//! let result = fixed_point(
//!     |x| array![x[0].cos()],
//!     array![0.0],
//!     FixedPointOptions { algorithm: Algorithm::Anderson, ..Default::default() },
//! );
//! assert_eq!(result.status, TerminationStatus::Converged);
//! ```

#![deny(missing_debug_implementations)]
#![warn(missing_docs)]

pub(crate) mod types;
pub(crate) mod acceleration;
pub(crate) mod iteration;

pub use iteration::{fixed_point, fixed_point_from};
pub use types::*;
