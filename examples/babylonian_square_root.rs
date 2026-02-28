use ndarray::Array1;
use fixed_point_acceleration::{fixed_point, Algorithm, FixedPointOptions};

fn main() {
    let target = 10.0;
    let babylonian = |x: &Array1<f64>| {
        let val = x[0];
        ndarray::array![0.5 * (val + target / val)]
    };

    let initial_guess = ndarray::array![1.0];
    let options = FixedPointOptions {
        algorithm: Algorithm::Newton,
        ..Default::default()
    };

    let result = fixed_point(babylonian, initial_guess, options);

    println!("Result: {:?}", result.outputs.last().unwrap());
    println!("Status: {:?}", result.status);
}
