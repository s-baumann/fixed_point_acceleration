use ndarray::Array1;
use crate::{FixedPointOptions, FixedPointResults};
use crate::ii_acceleration::get_new_input;

pub fn fixed_point<F>(
    func: F,
    x0: Array1<f64>,
    options: FixedPointOptions,
) -> FixedPointResults
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut convergence_vector: Vec<f64> = Vec::new();

    let mut current_input = x0;

    for iter in 0..options.max_iter {
        let current_output = func(&current_input);

        // Default Convergence Metric: L2 Norm of (Output - Input)
        let abs_residual: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = (&current_output - &current_input).mapv(|x| x.abs());
        let convergence: f64 = abs_residual.mean().expect("residual array is zero-length");
        inputs.push(current_input.clone());
        outputs.push(current_output.clone());
        convergence_vector.push(convergence);

        if options.print_reports {
            println!("Iter: {} | Convergence: {:.2e}", iter, convergence);
        }

        if convergence < options.threshold {
            return FixedPointResults {
                inputs, outputs, convergence_vector,
                status: "Reached Convergence Threshold".to_string(),
            };
        }

        // Generate next input using acceleration
        current_input = get_new_input(&inputs, &outputs, &options);
    }

    FixedPointResults {
        inputs, outputs, convergence_vector,
        status: "Reached Max Iterations".to_string(),
    }
}
