use ndarray::Array1;
use fixed_point_acceleration::{fixed_point, fixed_point_from, Algorithm, FixedPointOptions};

fn main() {
    let target = 10.0_f64;
    // g(x) = 0.5 * (x + 10/x) — the Babylonian / Heron's method for √10
    let babylonian = |x: &Array1<f64>| ndarray::array![0.5 * (x[0] + target / x[0])];

    // ── Single-algorithm run ──────────────────────────────────────────────────

    let result = fixed_point(
        babylonian,
        ndarray::array![1.0],
        FixedPointOptions { algorithm: Algorithm::Newton, ..Default::default() },
    );
    println!("=== Newton acceleration ===");
    println!("Result:     {:?}", result.outputs.last().unwrap()); // ≈ [3.16228]
    println!("Status:     {:?}", result.status);
    println!("Iterations: {}", result.inputs.len());

    // ── Algorithm switching mid-run ───────────────────────────────────────────
    //
    // fixed_point_from continues an existing result with a new algorithm.
    // The full history is preserved, and the new algorithm's counter resets to 1
    // so period-based methods (Aitken, Newton) and window sizes (Anderson, …)
    // are counted fresh from the switch point.

    // Phase 1: a few plain-substitution steps to move away from the initial guess.
    let warm = fixed_point(
        babylonian,
        ndarray::array![1.0],
        FixedPointOptions { algorithm: Algorithm::Simple, max_iter: 5, ..Default::default() },
    );
    println!("\n=== Simple (5 iters) then Anderson ===");
    println!("After Simple ({} iters): {:?}", warm.inputs.len(), warm.outputs.last().unwrap());

    // Phase 2: hand off to Anderson acceleration.
    let result2 = fixed_point_from(
        babylonian,
        warm,
        FixedPointOptions { algorithm: Algorithm::Anderson, ..Default::default() },
    );
    println!("Final result:     {:?}", result2.outputs.last().unwrap()); // ≈ [3.16228]
    println!("Status:           {:?}", result2.status);
    println!("Total iterations: {}", result2.inputs.len());
}
