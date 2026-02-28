/// Integration tests for fixed_point_acceleration.
///
/// Three test functions are used throughout:
///   - babylonian  : g(x) = 0.5*(x + 10/x),  fixed point √10 ≈ 3.16228   (scalar, fast)
///   - cosine      : g(x) = cos(x),            fixed point ≈ 0.73909        (scalar, slow)
///   - linear_3d   : g(x) = 0.5*x + [1,2,3],  fixed point [2, 4, 6]        (vector)
///
/// The cosine function is the most useful benchmark: it converges slowly under
/// Simple iteration (~65 steps to 1e-10), so acceleration methods show a clear benefit.
use fixed_point_acceleration::{fixed_point, fixed_point_from, Algorithm, FixedPointOptions, TerminationStatus};
use ndarray::{array, Array1};

// ── Known answers ─────────────────────────────────────────────────────────────

const SQRT10:  f64 = 3.162_277_660_168_379_5;
const DOTTIE:  f64 = 0.739_085_133_215_160_7;   // cos fixed point
const ANSWER_TOL: f64 = 1e-8;                    // tolerance for final-value checks

// ── Test function definitions ─────────────────────────────────────────────────

fn babylonian(x: &Array1<f64>) -> Array1<f64> {
    array![0.5 * (x[0] + 10.0 / x[0])]
}

fn cosine_fn(x: &Array1<f64>) -> Array1<f64> {
    array![x[0].cos()]
}

/// g(x) = 0.5·x + [1, 2, 3]  →  fixed point [2, 4, 6]
fn linear_3d(x: &Array1<f64>) -> Array1<f64> {
    x * 0.5 + array![1.0, 2.0, 3.0]
}

/// A non-contractive map so we can test the max-iter bail-out path.
fn diverging(x: &Array1<f64>) -> Array1<f64> {
    x * 2.0
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn opts(alg: Algorithm) -> FixedPointOptions {
    FixedPointOptions { algorithm: alg, ..Default::default() }
}

fn converged(status: &TerminationStatus) -> bool {
    *status == TerminationStatus::Converged
}

// ── Structural invariants ─────────────────────────────────────────────────────

/// inputs, outputs, and convergence_vector must always have the same length,
/// and that length must equal the number of iterations actually performed.
#[test]
fn result_lengths_are_consistent() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::Simple));
    let n = res.inputs.len();
    assert!(n > 0);
    assert_eq!(res.outputs.len(),           n, "outputs length mismatch");
    assert_eq!(res.convergence_vector.len(), n, "convergence_vector length mismatch");
}

/// convergence_vector entries must be non-negative (they are absolute residuals).
#[test]
fn convergence_vector_is_non_negative() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::Anderson));
    assert!(res.convergence_vector.iter().all(|&c| c >= 0.0));
}

/// When convergence is reached, the final convergence_vector entry must be
/// below the requested threshold.
#[test]
fn final_convergence_below_threshold() {
    let threshold = 1e-9;
    let res = fixed_point(
        cosine_fn,
        array![0.0],
        FixedPointOptions { algorithm: Algorithm::Simple, threshold, ..Default::default() },
    );
    assert!(converged(&res.status));
    assert!(*res.convergence_vector.last().unwrap() < threshold);
}

/// max_iter must be respected: if the function never converges, the loop stops
/// at max_iter and reports the correct status.
#[test]
fn max_iter_is_respected() {
    let max_iter = 20;
    let res = fixed_point(
        diverging,
        array![1.0],
        FixedPointOptions { algorithm: Algorithm::Simple, max_iter, ..Default::default() },
    );
    assert_eq!(res.status, TerminationStatus::MaxIterationsReached);
    assert_eq!(res.inputs.len(), max_iter);
}

// ── Scalar convergence: Babylonian √10 ───────────────────────────────────────

#[test]
fn simple_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::Simple));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn aitken_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::Aitken));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn newton_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::Newton));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn anderson_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::Anderson));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn mpe_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::MPE));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn rre_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::RRE));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn sea_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::SEA));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

#[test]
fn vea_babylonian() {
    let res = fixed_point(babylonian, array![1.0], opts(Algorithm::VEA));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - SQRT10).abs() < ANSWER_TOL);
}

// ── Scalar convergence: cosine Dottie number ─────────────────────────────────
//
// g(x) = cos(x) has spectral radius ≈ 0.674 at the fixed point, so Simple
// iteration needs ~65 steps.  Acceleration methods should need far fewer.

#[test]
fn simple_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::Simple));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn aitken_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::Aitken));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn newton_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::Newton));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn anderson_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::Anderson));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn mpe_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::MPE));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn rre_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::RRE));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn sea_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::SEA));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn vea_cosine() {
    let res = fixed_point(cosine_fn, array![0.0], opts(Algorithm::VEA));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

// ── Vector convergence: 3-element linear contraction ─────────────────────────

#[test]
fn simple_linear_3d() {
    let res = fixed_point(linear_3d, array![0.0, 0.0, 0.0], opts(Algorithm::Simple));
    assert!(converged(&res.status));
    let x = res.outputs.last().unwrap();
    assert!((x[0] - 2.0).abs() < ANSWER_TOL);
    assert!((x[1] - 4.0).abs() < ANSWER_TOL);
    assert!((x[2] - 6.0).abs() < ANSWER_TOL);
}

#[test]
fn anderson_linear_3d() {
    let res = fixed_point(linear_3d, array![0.0, 0.0, 0.0], opts(Algorithm::Anderson));
    assert!(converged(&res.status));
    let x = res.outputs.last().unwrap();
    assert!((x[0] - 2.0).abs() < ANSWER_TOL);
    assert!((x[1] - 4.0).abs() < ANSWER_TOL);
    assert!((x[2] - 6.0).abs() < ANSWER_TOL);
}

#[test]
fn mpe_linear_3d() {
    let res = fixed_point(linear_3d, array![0.0, 0.0, 0.0], opts(Algorithm::MPE));
    assert!(converged(&res.status));
    let x = res.outputs.last().unwrap();
    assert!((x[0] - 2.0).abs() < ANSWER_TOL);
    assert!((x[1] - 4.0).abs() < ANSWER_TOL);
    assert!((x[2] - 6.0).abs() < ANSWER_TOL);
}

#[test]
fn rre_linear_3d() {
    let res = fixed_point(linear_3d, array![0.0, 0.0, 0.0], opts(Algorithm::RRE));
    assert!(converged(&res.status));
    let x = res.outputs.last().unwrap();
    assert!((x[0] - 2.0).abs() < ANSWER_TOL);
    assert!((x[1] - 4.0).abs() < ANSWER_TOL);
    assert!((x[2] - 6.0).abs() < ANSWER_TOL);
}

#[test]
fn vea_linear_3d() {
    let res = fixed_point(linear_3d, array![0.0, 0.0, 0.0], opts(Algorithm::VEA));
    assert!(converged(&res.status));
    let x = res.outputs.last().unwrap();
    assert!((x[0] - 2.0).abs() < ANSWER_TOL);
    assert!((x[1] - 4.0).abs() < ANSWER_TOL);
    assert!((x[2] - 6.0).abs() < ANSWER_TOL);
}

// ── Acceleration effectiveness ────────────────────────────────────────────────
//
// On the slowly-converging cosine problem, every acceleration method should
// need fewer iterations than Simple to reach the same threshold.

fn cosine_iter_count(alg: Algorithm) -> usize {
    fixed_point(cosine_fn, array![0.0], opts(alg)).inputs.len()
}

#[test]
fn anderson_faster_than_simple_on_cosine() {
    let simple   = cosine_iter_count(Algorithm::Simple);
    let anderson = cosine_iter_count(Algorithm::Anderson);
    assert!(
        anderson < simple,
        "Anderson ({anderson} iters) should beat Simple ({simple} iters)",
    );
}

#[test]
fn mpe_faster_than_simple_on_cosine() {
    let simple = cosine_iter_count(Algorithm::Simple);
    let mpe    = cosine_iter_count(Algorithm::MPE);
    assert!(mpe < simple, "MPE ({mpe} iters) should beat Simple ({simple} iters)");
}

#[test]
fn sea_faster_than_simple_on_cosine() {
    let simple = cosine_iter_count(Algorithm::Simple);
    let sea    = cosine_iter_count(Algorithm::SEA);
    assert!(sea < simple, "SEA ({sea} iters) should beat Simple ({simple} iters)");
}

#[test]
fn vea_faster_than_simple_on_cosine() {
    let simple = cosine_iter_count(Algorithm::Simple);
    let vea    = cosine_iter_count(Algorithm::VEA);
    assert!(vea < simple, "VEA ({vea} iters) should beat Simple ({simple} iters)");
}

// ── Algorithm switching via fixed_point_from ──────────────────────────────────

#[test]
fn switch_simple_to_anderson_cosine() {
    // Warm up with 2 Simple iterations then hand off to Anderson.
    let warm = fixed_point(cosine_fn, array![0.0], FixedPointOptions {
        algorithm: Algorithm::Simple,
        max_iter:  2,
        ..Default::default()
    });
    // warm-up should NOT have converged (cosine takes ~65 Simple steps)
    assert!(!converged(&warm.status));

    let res = fixed_point_from(cosine_fn, warm, opts(Algorithm::Anderson));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}

#[test]
fn switch_preserves_full_history() {
    // The returned result should contain all iterates: prior + new.
    let warm = fixed_point(cosine_fn, array![0.0], FixedPointOptions {
        algorithm: Algorithm::Simple,
        max_iter:  3,
        ..Default::default()
    });
    let prior_len = warm.inputs.len();

    let res = fixed_point_from(cosine_fn, warm, opts(Algorithm::Anderson));
    assert!(res.inputs.len() > prior_len, "new iterates should be appended");
    assert_eq!(res.inputs.len(), res.outputs.len());
    assert_eq!(res.inputs.len(), res.convergence_vector.len());
}

#[test]
fn new_algorithm_k_resets() {
    // Aitken fires on iterations that are multiples-of-3 *within its own run*.
    // If we hand off after 4 Simple steps, Aitken should still converge
    // (k resets to 1 at the start of the Aitken run, so it fires at k=3,6,...).
    let warm = fixed_point(cosine_fn, array![0.0], FixedPointOptions {
        algorithm: Algorithm::Simple,
        max_iter:  4,
        ..Default::default()
    });
    let res = fixed_point_from(cosine_fn, warm, opts(Algorithm::Aitken));
    assert!(converged(&res.status));
    assert!((res.outputs.last().unwrap()[0] - DOTTIE).abs() < ANSWER_TOL);
}
