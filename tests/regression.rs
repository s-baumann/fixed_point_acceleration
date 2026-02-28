/// Regression tests for fixed_point_acceleration.
///
/// Reference values live in tests/regression_refs.yaml.
/// Initially seeded from the Julia implementation, they are maintained as
/// Rust regression baselines.  To regenerate after algorithm changes, run:
///
///   cargo test rebase_refs -- --ignored --nocapture
///
/// Test cases skipped at runtime:
///   - `is_complex: true`               — complex arithmetic not supported
///   - `dampening_with_input: true`     — not implemented
use fixed_point_acceleration::{fixed_point, fixed_point_from, Algorithm, FixedPointOptions, TerminationStatus};
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

#[derive(Deserialize)]
struct Options {
    #[serde(default = "one")]
    dampening: f64,
    #[serde(default)]
    dampening_with_input: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { dampening: 1.0, dampening_with_input: false }
    }
}

fn one() -> f64 { 1.0 }

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

/// Map a Rust TerminationStatus to the Julia termination symbol string.
fn rust_status_matches(status: &TerminationStatus, julia_status: &str) -> bool {
    let normalised = match status {
        TerminationStatus::Converged => "ReachedConvergenceThreshold",
        TerminationStatus::MaxIterationsReached => "ReachedMaxIter",
    };
    normalised == julia_status
}

fn alg_from_str(name: &str) -> Algorithm {
    match name {
        "Simple"   => Algorithm::Simple,
        "Anderson" => Algorithm::Anderson,
        "Aitken"   => Algorithm::Aitken,
        "Newton"   => Algorithm::Newton,
        "MPE"      => Algorithm::MPE,
        "RRE"      => Algorithm::RRE,
        "VEA"      => Algorithm::VEA,
        "SEA"      => Algorithm::SEA,
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

// ── Algorithm-switching regression ────────────────────────────────────────────
//
// Pins the exact behaviour of fixed_point_from:
//   - the prior history is present verbatim at the start of the combined result
//   - the new algorithm's k counter resets so period-based methods fire correctly
//   - the combined run converges to the known fixed point
//
// Test function: g(x) = cos(x), fixed point (Dottie constant) ≈ 0.739085133215
// Initial guess: x = [0.0]
//
// The first three Simple iterates are exact in f64:
//   iter 0: input = 0.0,            output = cos(0.0)            = 1.0
//   iter 1: input = 1.0,            output = cos(1.0)            (IEEE-deterministic)
//   iter 2: input = cos(1.0),       output = cos(cos(1.0))       (IEEE-deterministic)

#[test]
fn algorithm_switch_simple_to_anderson() {
    let g = |x: &Array1<f64>| x.mapv(f64::cos);

    // ── Phase 1: 3 Simple iterations ─────────────────────────────────────────
    let warm = fixed_point(
        g,
        array![0.0],
        FixedPointOptions {
            algorithm: Algorithm::Simple,
            max_iter: 3,
            ..Default::default()
        },
    );

    assert_eq!(warm.inputs.len(), 3);
    assert_eq!(warm.outputs.len(), 3);

    // Pin exact Simple iterates (IEEE-deterministic).
    let cos1     = (1.0_f64).cos();
    let cos_cos1 = cos1.cos();
    assert_eq!(warm.inputs[0][0],  0.0);
    assert_eq!(warm.outputs[0][0], 1.0);
    assert_eq!(warm.inputs[1][0],  1.0);
    assert_eq!(warm.outputs[1][0], cos1);
    assert_eq!(warm.inputs[2][0],  cos1);
    assert_eq!(warm.outputs[2][0], cos_cos1);

    // ── Phase 2: continue with Anderson ──────────────────────────────────────
    let result = fixed_point_from(
        g,
        warm,
        FixedPointOptions {
            algorithm: Algorithm::Anderson,
            ..Default::default()
        },
    );

    // Combined history begins with the exact prior iterates.
    assert_eq!(result.inputs[0][0],  0.0);
    assert_eq!(result.outputs[0][0], 1.0);
    assert_eq!(result.inputs[1][0],  1.0);
    assert_eq!(result.outputs[1][0], cos1);
    assert_eq!(result.inputs[2][0],  cos1);
    assert_eq!(result.outputs[2][0], cos_cos1);

    // The first input of the Anderson phase is the last output of the warm-up.
    assert_eq!(result.inputs[3][0], cos_cos1);

    // Converged to the Dottie constant.
    assert_eq!(result.status, TerminationStatus::Converged);
    let fp = result.outputs.last().unwrap()[0];
    assert!((fp - 0.739_085_133_215_160_7).abs() < 1e-10,
        "fixed point {fp} too far from Dottie constant");

    // History is self-consistent throughout.
    assert_eq!(result.inputs.len(), result.outputs.len());
    assert_eq!(result.inputs.len(), result.convergence_vector.len());
    assert!(result.inputs.len() > 3, "Anderson should have added iterations");
}

// ── YAML regression ───────────────────────────────────────────────────────────

fn yaml_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/regression_refs.yaml")
}

#[test]
fn cross_language_regression() {
    let yaml_content = std::fs::read_to_string(yaml_path()).unwrap_or_else(|e| {
        panic!("Could not read regression_refs.yaml: {e}")
    });

    let refs: Refs =
        serde_yaml::from_str(&yaml_content).expect("Failed to parse regression_refs.yaml");

    for tc in &refs.test_cases {
        if tc.is_complex { continue; }
        if tc.options.dampening_with_input { continue; }

        let x0 = tc.x0.as_array().expect("x0 must be real");
        let iter_tol = tc.iterate_tol.unwrap_or(1e-12);

        for (alg_name, alg_spec) in &tc.algorithms {
            if !alg_spec.expected_fixed_point.is_real() { continue; }

            let label = format!("{}/{}", tc.name, alg_name);
            let alg   = alg_from_str(alg_name);
            let opts  = FixedPointOptions {
                algorithm: alg,
                threshold: tc.convergence_threshold,
                max_iter:  tc.max_iter,
                dampening: tc.options.dampening,
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
                result.status, alg_spec.expected_termination,
            );

            // ── Fixed point (last output) ─────────────────────────────────────
            let fp = result.outputs.last().expect("no outputs produced");
            let expected_fp = alg_spec.expected_fixed_point.as_array().unwrap();
            assert_vec_approx(fp, &expected_fp, tc.fixed_point_tol,
                &format!("{label} fixed_point"));

            // ── First N iterates ──────────────────────────────────────────────
            let n_check = alg_spec.iterates.len().min(result.inputs.len());
            for (i, iter_ref) in alg_spec.iterates.iter().take(n_check).enumerate() {
                let exp_in  = iter_ref.input.as_array().expect("iterate input is real");
                let exp_out = iter_ref.output.as_array().expect("iterate output is real");
                assert_vec_approx(&result.inputs[i],  &exp_in,  iter_tol,
                    &format!("{label} input[{i}]"));
                assert_vec_approx(&result.outputs[i], &exp_out, iter_tol,
                    &format!("{label} output[{i}]"));
            }
        }
    }
}

// ── Reference generator ───────────────────────────────────────────────────────
//
// Overwrites regression_refs.yaml with values produced by this Rust
// implementation.  Run once whenever algorithm behaviour changes:
//
//   cargo test rebase_refs -- --ignored --nocapture

#[test]
#[ignore]
fn rebase_refs() {
    use serde_yaml::{Mapping, Number, Value};

    let path = yaml_path();
    let content = std::fs::read_to_string(&path)
        .expect("Could not read regression_refs.yaml");
    let mut root: Value = serde_yaml::from_str(&content)
        .expect("Failed to parse regression_refs.yaml");

    let test_cases = root["test_cases"].as_sequence_mut()
        .expect("test_cases must be a sequence");

    for tc_val in test_cases.iter_mut() {
        // Skip cases we cannot run.
        if tc_val["is_complex"].as_bool().unwrap_or(false) { continue; }
        let dampening_with_input = tc_val.get("options")
            .and_then(|o| o.get("dampening_with_input"))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        if dampening_with_input { continue; }

        // Read test-case parameters (immutable phase — drop borrows before mutating).
        let func_name: String = tc_val["function"].as_str().unwrap().to_string();
        let x0_vals: Vec<f64> = tc_val["x0"].as_sequence().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        let x0        = Array1::from_vec(x0_vals);
        let threshold = tc_val["convergence_threshold"].as_f64().unwrap();
        let max_iter  = tc_val["max_iter"].as_u64().unwrap() as usize;
        let dampening = tc_val.get("options")
            .and_then(|o| o.get("dampening"))
            .and_then(|d| d.as_f64())
            .unwrap_or(1.0);
        let alg_names: Vec<String> = tc_val["algorithms"].as_mapping().unwrap()
            .keys()
            .filter_map(|k| k.as_str().map(str::to_string))
            .collect();

        // Run all algorithms (no borrows from tc_val).
        let results: Vec<(String, _)> = alg_names.iter().map(|alg_name| {
            let alg  = alg_from_str(alg_name);
            let opts = FixedPointOptions {
                algorithm: alg,
                threshold,
                max_iter,
                dampening,
                ..Default::default()
            };
            let fn2 = func_name.clone();
            let res = fixed_point(
                move |x: &Array1<f64>| call_function(&fn2, x),
                x0.clone(),
                opts,
            );
            (alg_name.clone(), res)
        }).collect();

        // Helper: build a YAML sequence of floats.
        let make_seq = |vals: &[f64]| -> Value {
            Value::Sequence(
                vals.iter().map(|&v| Value::Number(Number::from(v))).collect()
            )
        };

        // Write results back into tc_val (mutable phase).
        let alg_map = tc_val["algorithms"].as_mapping_mut().unwrap();
        for (alg_name, result) in results {
            let alg_entry = alg_map
                .get_mut(Value::String(alg_name.clone()))
                .unwrap_or_else(|| panic!("algorithm {alg_name} not in YAML"));

            // First 4 iterates.
            let n = result.inputs.len().min(4);
            let iterates: Vec<Value> = (0..n).map(|i| {
                let mut m = Mapping::new();
                m.insert(
                    Value::String("input".into()),
                    make_seq(&result.inputs[i].to_vec()),
                );
                m.insert(
                    Value::String("output".into()),
                    make_seq(&result.outputs[i].to_vec()),
                );
                Value::Mapping(m)
            }).collect();

            let fixed_pt = make_seq(&result.outputs.last().unwrap().to_vec());
            let status   = match result.status {
                TerminationStatus::Converged          => "ReachedConvergenceThreshold",
                TerminationStatus::MaxIterationsReached => "ReachedMaxIter",
            };

            alg_entry["iterates"]              = Value::Sequence(iterates);
            alg_entry["expected_fixed_point"]  = fixed_pt;
            alg_entry["expected_termination"]  = Value::String(status.to_string());
        }
    }

    let new_yaml = serde_yaml::to_string(&root)
        .expect("Failed to serialise updated YAML");
    std::fs::write(&path, new_yaml)
        .expect("Failed to write regression_refs.yaml");
    println!("Rebased: {}", path.display());
}
