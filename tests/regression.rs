/// Cross-language regression tests for fixed_point_acceleration.
///
/// Reference values are read from the shared YAML file that is also used by
/// the Julia implementation.  Both languages must produce the same:
///   - termination status
///   - fixed point (final output vector)
///   - inputs and outputs at the first four iterates
///
/// Test cases flagged `is_complex: true` are skipped because this Rust crate
/// does not yet support complex arithmetic.
/// Test cases with `options.dampening != 1.0` are skipped because dampening
/// is not yet applied in the Rust iteration loop.
///
/// Shared YAML location: tests/regression_refs.yaml
/// This is a symlink to the Julia project's canonical copy:
///   ../../../library_julia/FixedPointAcceleration.jl/test/regression_refs.yaml
/// Julia's rebase command writes the reference values there; both test suites
/// then read the same file.

use fixed_point_acceleration::{fixed_point, Algorithm, FixedPointOptions};
use ndarray::{array, Array1};
use serde::Deserialize;
use std::collections::HashMap;

// ── YAML schema ───────────────────────────────────────────────────────────────

/// A vector that may be real (plain list) or complex ({real, imag} dict).
/// Complex entries are skipped at runtime; the enum lets serde parse both.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum VecSpec {
    Real(Vec<f64>),
    // Fields are parsed from YAML but only used for the is_real() check.
    #[allow(dead_code)]
    Complex { real: Vec<f64>, imag: Vec<f64> },
}

impl VecSpec {
    fn is_real(&self) -> bool {
        matches!(self, VecSpec::Real(_))
    }

    fn as_array(&self) -> Option<Array1<f64>> {
        match self {
            VecSpec::Real(v) => Some(Array1::from_vec(v.clone())),
            VecSpec::Complex { .. } => None,
        }
    }
}

#[derive(Deserialize)]
struct IterateRef {
    input: VecSpec,
    output: VecSpec,
}

#[derive(Deserialize)]
struct AlgExpected {
    expected_termination: String,
    expected_fixed_point: VecSpec,
    #[serde(default)]
    iterates: Vec<IterateRef>,
}

#[derive(Deserialize, Default)]
struct Options {
    #[serde(default = "one")]
    dampening: f64,
    #[serde(default)]
    dampening_with_input: bool,
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    function: String,
    x0: VecSpec,
    is_complex: bool,
    max_iter: usize,
    convergence_threshold: f64,
    fixed_point_tol: f64,
    #[serde(default)]
    iterate_tol: Option<f64>,
    #[serde(default)]
    options: Options,
    algorithms: HashMap<String, AlgExpected>,
}

#[derive(Deserialize)]
struct Refs {
    test_cases: Vec<TestCase>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Map a Julia termination symbol string to the Rust status string.
fn rust_status_matches(rust_status: &str, julia_status: &str) -> bool {
    let normalised = match rust_status {
        "Reached Convergence Threshold" => "ReachedConvergenceThreshold",
        "Reached Max Iterations" => "ReachedMaxIter",
        other => other,
    };
    normalised == julia_status
}

fn alg_from_str(name: &str) -> Algorithm {
    match name {
        "Simple" => Algorithm::Simple,
        "Anderson" => Algorithm::Anderson,
        "Aitken" => Algorithm::Aitken,
        "Newton" => Algorithm::Newton,
        "MPE" => Algorithm::MPE,
        "RRE" => Algorithm::RRE,
        "VEA" => Algorithm::VEA,
        "SEA" => Algorithm::SEA,
        other => panic!("Unknown algorithm: {other}"),
    }
}

fn assert_vec_approx(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch (got {}, expected {})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: got {a:.16e}, expected {e:.16e}, diff {diff:.3e}, tol {tol:.3e}",
        );
    }
}

// ── Function registry ─────────────────────────────────────────────────────────
// Keys must match those used in the YAML `function` field and in the Julia
// FUNC_REGISTRY in test/RegressionTest.jl.

fn call_function(name: &str, x: &Array1<f64>) -> Array1<f64> {
    match name {
        "cos" => x.mapv(f64::cos),
        "vector_2d" => array![
            0.5 * (x[0] + x[1]).abs().sqrt(),
            1.5 * x[0] + 0.5 * x[1]
        ],
        "sqrt" => x.mapv(f64::sqrt),
        other => panic!("Unknown function in registry: {other}"),
    }
}

// ── Test ─────────────────────────────────────────────────────────────────────

#[test]
fn cross_language_regression() {
    let yaml_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/regression_refs.yaml");

    let yaml_content = std::fs::read_to_string(&yaml_path).unwrap_or_else(|e| {
        panic!(
            "Could not read tests/regression_refs.yaml at {}: {e}\n\
             The file is a symlink to the Julia project's regression_refs.yaml.\n\
             Run Julia rebase to regenerate it:\n\
             julia -e 'using Pkg; withenv(\"REBASE_REFS\" => \"true\") do; Pkg.test(); end'",
            yaml_path.display()
        )
    });

    let refs: Refs =
        serde_yaml::from_str(&yaml_content).expect("Failed to parse regression_refs.yaml");

    for tc in &refs.test_cases {
        // Skip test cases this implementation cannot yet handle.
        if tc.is_complex {
            continue; // no complex support
        }
        if (tc.options.dampening - 1.0).abs() > 1e-15 || tc.options.dampening_with_input {
            continue; // dampening not implemented
        }

        let x0 = tc
            .x0
            .as_array()
            .expect("x0 must be real for non-complex test cases");

        let iter_tol = tc.iterate_tol.unwrap_or(1e-12);

        for (alg_name, alg_spec) in &tc.algorithms {
            // Skip if the stored fixed point is complex (shouldn't happen for
            // non-complex test cases, but guard anyway).
            if !alg_spec.expected_fixed_point.is_real() {
                continue;
            }

            let label = format!("{}/{}", tc.name, alg_name);

            let alg = alg_from_str(alg_name);
            let opts = FixedPointOptions {
                algorithm: alg,
                threshold: tc.convergence_threshold,
                max_iter: tc.max_iter,
                ..Default::default()
            };

            let func_name = tc.function.clone();
            let result = fixed_point(
                move |x: &Array1<f64>| call_function(&func_name, x),
                x0.clone(),
                opts,
            );

            // ── Termination status ────────────────────────────────────────────
            assert!(
                rust_status_matches(&result.status, &alg_spec.expected_termination),
                "{label}: status mismatch — got {:?}, expected {:?}",
                result.status,
                alg_spec.expected_termination,
            );

            // ── Fixed point (last output) ──────────────────────────────────────
            let fp = result.outputs.last().expect("no outputs produced");
            let expected_fp = alg_spec
                .expected_fixed_point
                .as_array()
                .expect("expected_fixed_point should be real");
            assert_vec_approx(fp, &expected_fp, tc.fixed_point_tol, &format!("{label} fixed_point"));

            // ── First N iterates ───────────────────────────────────────────────
            let n_check = alg_spec.iterates.len().min(result.inputs.len());
            for (i, iter_ref) in alg_spec.iterates.iter().take(n_check).enumerate() {
                let exp_input = iter_ref
                    .input
                    .as_array()
                    .expect("iterate input should be real");
                let exp_output = iter_ref
                    .output
                    .as_array()
                    .expect("iterate output should be real");

                assert_vec_approx(
                    &result.inputs[i],
                    &exp_input,
                    iter_tol,
                    &format!("{label} input[{i}]"),
                );
                assert_vec_approx(
                    &result.outputs[i],
                    &exp_output,
                    iter_tol,
                    &format!("{label} output[{i}]"),
                );
            }
        }
    }
}
