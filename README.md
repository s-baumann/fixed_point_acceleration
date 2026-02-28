# fixed_point_acceleration

[![CI](https://github.com/s-baumann/fixed_point_acceleration/actions/workflows/ci.yml/badge.svg)](https://github.com/s-baumann/fixed_point_acceleration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/s-baumann/fixed_point_acceleration/graph/badge.svg?token=tJASwGGy4l)](https://codecov.io/gh/s-baumann/fixed_point_acceleration)
[![crates.io](https://img.shields.io/crates/v/fixed_point_acceleration.svg)](https://crates.io/crates/fixed_point_acceleration)

A Rust library for finding fixed points of vector-valued functions using acceleration algorithms.

Given a function `g: ℝⁿ → ℝⁿ`, finds `x*` such that `g(x*) = x*`. Acceleration methods reduce the number of iterations required compared to plain successive substitution, often dramatically.

## Usage

```rust
use fixed_point_acceleration::{fixed_point, Algorithm, FixedPointOptions};
use ndarray::array;

// Find √10 via the Babylonian method: g(x) = 0.5 * (x + 10/x)
let result = fixed_point(
    |x| array![0.5 * (x[0] + 10.0 / x[0])],
    array![1.0],                              // initial guess
    FixedPointOptions {
        algorithm: Algorithm::Anderson,
        ..Default::default()
    },
);

println!("Result: {:?}", result.outputs.last().unwrap()); // ≈ [3.16228]
println!("Status: {}",   result.status);                 // Converged
println!("Iterations: {}", result.inputs.len());
```

## Switching algorithms mid-run

`fixed_point_from` continues an existing result with a new algorithm. The full history is preserved in the returned result, and the new algorithm's iteration counter resets to 1 so period-based methods (Aitken, Newton) and window sizes (Anderson, MPE, …) behave correctly from the switch point.

```rust
use fixed_point_acceleration::{fixed_point, fixed_point_from, Algorithm, FixedPointOptions};
use ndarray::array;

let g = |x: &ndarray::Array1<f64>| array![x[0].cos()]; // fixed point ≈ 0.73909

// Phase 1: a few plain-substitution steps to get away from the initial guess
let warm = fixed_point(
    g,
    array![0.0],
    FixedPointOptions {
        algorithm: Algorithm::Simple,
        max_iter:  5,
        ..Default::default()
    },
);

// Phase 2: hand off to Anderson — k resets to 1 at the switch point
let result = fixed_point_from(
    g,
    warm,
    FixedPointOptions {
        algorithm: Algorithm::Anderson,
        ..Default::default()
    },
);

// result contains the combined history from both phases
println!("Total iterations: {}", result.inputs.len());
println!("Result: {:?}", result.outputs.last().unwrap()); // ≈ [0.73909]
println!("Status: {}",   result.status);                  // Converged
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `Simple`   | Successive substitution (Picard iteration) |
| `Aitken`   | Aitken's Δ² extrapolation; applied every 3rd iteration |
| `Newton`   | Secant-based acceleration; applied every 3rd iteration |
| `Anderson` | Anderson mixing over a window of recent iterates |
| `MPE`      | Minimal Polynomial Extrapolation |
| `RRE`      | Reduced Rank Extrapolation |
| `SEA`      | Scalar Epsilon Algorithm (Wynn ε, element-wise) |
| `VEA`      | Vector Epsilon Algorithm (Wynn ε, pseudoinverse) |

`MPE`, `RRE`, `SEA`, and `VEA` are applied periodically every `extrapolation_period` iterations, using simple substitution in between.

## Options

```rust
FixedPointOptions {
    algorithm:            Algorithm::Anderson, // acceleration method
    threshold:            1e-10,               // convergence: mean |g(x) - x|
    max_iter:             1000,                // hard iteration cap
    max_m:                10,                  // history window (Anderson/MPE/RRE/SEA/VEA)
    extrapolation_period: 7,                   // how often MPE/RRE/SEA/VEA fire
    dampening:            1.0,                 // step-size damping (1.0 = none)
}
```

## Results

```rust
pub struct FixedPointResults {
    pub inputs:             Vec<Array1<f64>>, // input  at each iteration
    pub outputs:            Vec<Array1<f64>>, // g(x)   at each iteration
    pub convergence_vector: Vec<f64>,         // mean |g(x) - x| at each iteration
    pub status:             TerminationStatus, // Converged | MaxIterationsReached | NumericalFailure | FunctionFailure
}
```

