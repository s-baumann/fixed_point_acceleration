# fixed_point_acceleration

[![CI](https://github.com/s-baumann/fixed_point_acceleration/actions/workflows/ci.yml/badge.svg)](https://github.com/s-baumann/fixed_point_acceleration/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/s-baumann/fixed_point_acceleration/branch/main/graph/badge.svg)](https://codecov.io/gh/s-baumann/fixed_point_acceleration)
[![crates.io](https://img.shields.io/crates/v/fixed_point_acceleration.svg)](https://crates.io/crates/fixed_point_acceleration)
[![docs.rs](https://docs.rs/fixed_point_acceleration/badge.svg)](https://docs.rs/fixed_point_acceleration)

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
println!("Status: {}",   result.status);
println!("Iterations: {}", result.inputs.len());
```

## Algorithms

| Algorithm | Description | Minimum iterates |
|-----------|-------------|-----------------|
| `Simple`   | Successive substitution (Picard iteration) | 1 |
| `Aitken`   | Aitken's Δ² extrapolation; applied every 3rd iteration | 3 |
| `Newton`   | Secant-based acceleration; applied every 3rd iteration | 2 |
| `Anderson` | Anderson mixing over a window of recent iterates | 2 |
| `MPE`      | Minimal Polynomial Extrapolation | 3 |
| `RRE`      | Reduced Rank Extrapolation | 4 |
| `SEA`      | Scalar Epsilon Algorithm (Wynn ε, element-wise) | 3 |
| `VEA`      | Vector Epsilon Algorithm (Wynn ε, pseudoinverse) | 3 |

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
    print_reports:        false,               // print convergence each iteration
}
```

## Results

```rust
pub struct FixedPointResults {
    pub inputs:             Vec<Array1<f64>>, // input  at each iteration
    pub outputs:            Vec<Array1<f64>>, // g(x)   at each iteration
    pub convergence_vector: Vec<f64>,         // mean |g(x) - x| at each iteration
    pub status:             String,           // "Reached Convergence Threshold" | "Reached Max Iterations"
}
```

## System requirements

The linear algebra backend (`ndarray-linalg`) links against the system LAPACK and BLAS libraries. Install them before building:

```bash
# Debian / Ubuntu
sudo apt-get install libblas-dev liblapack-dev libgfortran5
```

On Debian/Ubuntu, CBLAS is merged into `libblas.so` rather than shipped as a separate file. The repository includes a `.cargo/config.toml` that points the linker at local shim symlinks. Create them once after cloning:

```bash
mkdir -p .cargo/lib
ln -sf /usr/lib/x86_64-linux-gnu/libblas.so  .cargo/lib/libcblas.so
ln -sf /usr/lib/x86_64-linux-gnu/liblapack.so .cargo/lib/liblapacke.so
```
