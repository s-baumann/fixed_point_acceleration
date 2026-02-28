use ndarray::Array1;
use crate::{FixedPointOptions, FixedPointResults, TerminationStatus};
use crate::acceleration::get_new_input;

/// Find a fixed point of `func` starting from `x0`.
pub fn fixed_point<F>(
    func: F,
    x0: Array1<f64>,
    options: FixedPointOptions,
) -> FixedPointResults
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    run_iterations(func, x0, Vec::new(), Vec::new(), Vec::new(), options)
}

/// Continue fixed-point iteration from a previous result with a new algorithm.
///
/// The full history from `prior` is preserved in the returned result, but the
/// new algorithm sees only the iterates produced during this call — so
/// period-based checks (Aitken, Newton) and window sizes (Anderson, MPE, …)
/// are counted fresh from the moment the algorithm switches.
pub fn fixed_point_from<F>(
    func: F,
    prior: FixedPointResults,
    options: FixedPointOptions,
) -> FixedPointResults
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let x0 = prior.outputs.last().expect("prior result has no outputs").clone();
    run_iterations(func, x0, prior.inputs, prior.outputs, prior.convergence_vector, options)
}

// Shared iteration loop.
//
// `acc_inputs`, `acc_outputs`, `acc_convergence` carry any history inherited
// from a prior run.  The new algorithm's `k` counter starts at 1 on the first
// new iterate because `get_new_input` receives only the slice that was
// appended during *this* call.
fn run_iterations<F>(
    func: F,
    x0: Array1<f64>,
    mut inputs: Vec<Array1<f64>>,
    mut outputs: Vec<Array1<f64>>,
    mut convergence_vector: Vec<f64>,
    options: FixedPointOptions,
) -> FixedPointResults
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let prior_len = inputs.len(); // index where this algorithm's iterates begin
    let mut current_input = x0;

    for iter in 0..options.max_iter {
        let current_output = func(&current_input);

        if !current_output.iter().all(|v| v.is_finite()) {
            return FixedPointResults {
                inputs, outputs, convergence_vector,
                status: TerminationStatus::FunctionFailure,
            };
        }

        let convergence: f64 = (&current_output - &current_input)
            .mapv(|x| x.abs())
            .mean()
            .expect("residual array is zero-length");

        inputs.push(current_input.clone());
        outputs.push(current_output.clone());
        convergence_vector.push(convergence);

        log::debug!("Iter: {} | Convergence: {:.2e}", iter, convergence);

        if convergence < options.threshold {
            return FixedPointResults {
                inputs, outputs, convergence_vector,
                status: TerminationStatus::Converged,
            };
        }

        // Slice to only this algorithm's history so k counts from 1.
        // None means the acceleration step produced a non-finite value.
        let Some(proposed) = get_new_input(&inputs[prior_len..], &outputs[prior_len..], &options)
        else {
            return FixedPointResults {
                inputs, outputs, convergence_vector,
                status: TerminationStatus::NumericalFailure,
            };
        };
        current_input = if options.dampening == 1.0 {
            proposed
        } else {
            inputs.last().unwrap() * (1.0 - options.dampening) + &proposed * options.dampening
        };
    }

    FixedPointResults {
        inputs, outputs, convergence_vector,
        status: TerminationStatus::MaxIterationsReached,
    }
}
