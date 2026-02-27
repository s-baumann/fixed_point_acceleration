use ndarray::{Array1, Array2, s};
use linfa_linalg::svd::SVD;
use crate::{Algorithm, FixedPointOptions};

pub(crate) fn get_new_input(
    inputs: &[Array1<f64>],
    outputs: &[Array1<f64>],
    options: &FixedPointOptions,
) -> Array1<f64> {
    let k = inputs.len();
    let last_output = outputs.last().unwrap().clone();

    match options.algorithm {
        Algorithm::Simple => last_output,

        Algorithm::Aitken => {
            if !k.is_multiple_of(3) { return last_output; }
            let x   = &inputs[k-2];
            let fx  = &outputs[k-2];
            let ffx = &outputs[k-1];

            let numerator   = (fx - x).mapv(|v: f64| v.powi(2));
            let denominator = ffx - fx * 2.0 + x;
            x - numerator / denominator
        },

        Algorithm::Newton => {
            if !k.is_multiple_of(3) { return last_output; }
            let xk1  = &inputs[k-2];
            let fxk1 = &outputs[k-2];
            let gxk1 = fxk1 - xk1;
            let xk   = &inputs[k-1];
            let fxk  = &outputs[k-1];
            let gxk  = fxk - xk;
            // Note: if x is a vector this is applied element-wise
            let derivative = (&gxk - &gxk1) / (xk - xk1);
            xk - (gxk / derivative)
        },

        Algorithm::Anderson => {
            let m = std::cmp::min(k, options.max_m);
            if m < 2 { return last_output; }
            perform_anderson(inputs, outputs, m)
        },

        // Polynomial extrapolation methods — applied every extrapolation_period iterations
        Algorithm::MPE => {
            if !k.is_multiple_of(options.extrapolation_period) { return last_output; }
            let m = std::cmp::min(k, options.max_m);
            if m < 3 { return last_output; }
            perform_mpe(outputs, m).unwrap_or(last_output)
        },

        Algorithm::RRE => {
            if !k.is_multiple_of(options.extrapolation_period) { return last_output; }
            let m = std::cmp::min(k, options.max_m);
            if m < 4 { return last_output; }
            perform_rre(outputs, m).unwrap_or(last_output)
        },

        // Epsilon extrapolation methods — applied every extrapolation_period iterations
        Algorithm::SEA => {
            if !k.is_multiple_of(options.extrapolation_period) { return last_output; }
            let m = std::cmp::min(k, options.max_m);
            if m < 3 { return last_output; }
            perform_epsilon(outputs, m, true).unwrap_or(last_output)
        },

        Algorithm::VEA => {
            if !k.is_multiple_of(options.extrapolation_period) { return last_output; }
            let m = std::cmp::min(k, options.max_m);
            if m < 3 { return last_output; }
            perform_epsilon(outputs, m, false).unwrap_or(last_output)
        },
    }
}

// Moore-Penrose pseudoinverse via SVD: pinv(A) = V * diag(s⁺) * Uᵀ
//
// ndarray-linalg returns the *full* U (m×m) and Vt (n×n) even for non-square
// matrices, while Sigma has only k = min(m,n) entries. We slice both down to
// the k-column/k-row thin portion before computing the product.
fn pseudoinverse(a: &Array2<f64>) -> Option<Array2<f64>> {
    let (m, n) = a.dim();
    let k = m.min(n);

    let (u_opt, s, vt_opt) = a.svd(true, true).ok()?;
    let u  = u_opt?;
    let vt = vt_opt?;

    let max_s = s.iter().cloned().fold(0.0_f64, f64::max);
    let tol   = 1e-10 * max_s;
    let s_inv = s.mapv(|sv| if sv > tol { 1.0 / sv } else { 0.0 });

    // Thin portions: U_thin is m×k, Vt_thin is k×n
    let u_thin  = u.slice(s![.., ..k]).to_owned();
    let vt_thin = vt.slice(s![..k, ..]).to_owned();

    // V * diag(s_inv) * Uᵀ  →  (n×k) @ (k×m)  →  n×m
    let mut vs = vt_thin.t().to_owned(); // n × k
    for (j, &si) in s_inv.iter().enumerate() {
        vs.column_mut(j).mapv_inplace(|v| v * si);
    }
    Some(vs.dot(&u_thin.t()))
}

// Build an n × m matrix whose columns are the last m entries of `outputs`
fn build_iterates_matrix(outputs: &[Array1<f64>], m: usize) -> Array2<f64> {
    let k = outputs.len();
    let n = outputs[0].len();
    let mut mat = Array2::<f64>::zeros((n, m));
    for (j, i) in (k - m..k).enumerate() {
        mat.column_mut(j).assign(&outputs[i]);
    }
    mat
}

fn is_finite_array(a: &Array1<f64>) -> bool {
    a.iter().all(|v| v.is_finite())
}

// Anderson acceleration (unchanged from original)
fn perform_anderson(inputs: &[Array1<f64>], outputs: &[Array1<f64>], m: usize) -> Array1<f64> {
    let k = inputs.len();
    let start = k - m;

    let residuals: Vec<Array1<f64>> = (start..k)
        .map(|i| &outputs[i] - &inputs[i])
        .collect();

    let last_residual = residuals.last().unwrap();
    let n = residuals[0].len();

    let mut delta_f = Array2::<f64>::zeros((n, m - 1));
    for (j, residual) in residuals.iter().enumerate().take(m - 1) {
        let col = residual - last_residual;
        delta_f.column_mut(j).assign(&col);
    }

    let lhs = delta_f.t().dot(&delta_f);
    let rhs = -delta_f.t().dot(last_residual);

    let c = match pseudoinverse(&lhs) {
        Some(pinv) => pinv.dot(&rhs),
        None => return outputs.last().unwrap().clone(),
    };
    if !is_finite_array(&c) { return outputs.last().unwrap().clone(); }

    let c_sum: f64 = c.sum();
    let last_output = &outputs[k - 1];
    let mut result = last_output * (1.0 - c_sum);
    for (j, &cj) in c.iter().enumerate() {
        result = result + &outputs[start + j] * cj;
    }
    result
}

// MPE — Minimal Polynomial Extrapolation
//
// Given m iterates as columns of U (n × m):
//   old_diffs = first m-2 consecutive differences        (n × m-2)
//   last_diff = final consecutive difference              (n)
//   c         = [-pinv(old_diffs) @ last_diff; 1]        (m-1)
//   result    = (U[:,1:] @ c) / sum(c)
fn perform_mpe(outputs: &[Array1<f64>], m: usize) -> Option<Array1<f64>> {
    let iterates = build_iterates_matrix(outputs, m); // n × m
    let n = iterates.nrows();

    // Build old_diffs (n × m-2) and last_diff (n)
    let mut old_diffs = Array2::<f64>::zeros((n, m - 2));
    for j in 0..m - 2 {
        old_diffs.column_mut(j).assign(
            &(iterates.column(j + 1).to_owned() - iterates.column(j))
        );
    }
    let last_diff = iterates.column(m - 1).to_owned() - iterates.column(m - 2);

    // c = [-pinv(old_diffs) @ last_diff, 1]
    let c_head = -pseudoinverse(&old_diffs)?.dot(&last_diff); // length m-2
    let mut c = Array1::<f64>::zeros(m - 1);
    c.slice_mut(s![..m - 2]).assign(&c_head);
    c[m - 2] = 1.0;

    let sum_c = c.sum();
    if sum_c.abs() < 1e-12 { return None; }

    // result = U[:,1:] @ c / sum_c
    let result = iterates.slice(s![.., 1..]).dot(&c) / sum_c;
    if !is_finite_array(&result) { return None; }
    Some(result)
}

// RRE — Reduced Rank Extrapolation
//
// Given m iterates as columns of U (n × m):
//   D1        = consecutive differences of U             (n × m-1)
//   D2        = consecutive differences of D1            (n × m-2)
//   result    = U[:,0] - D1[:,0:m-2] @ (pinv(D2) @ D1[:,0])
fn perform_rre(outputs: &[Array1<f64>], m: usize) -> Option<Array1<f64>> {
    let iterates = build_iterates_matrix(outputs, m); // n × m
    let n = iterates.nrows();

    // First differences D1 (n × m-1)
    let mut d1 = Array2::<f64>::zeros((n, m - 1));
    for j in 0..m - 1 {
        d1.column_mut(j).assign(
            &(iterates.column(j + 1).to_owned() - iterates.column(j))
        );
    }

    // Second differences D2 (n × m-2)
    let mut d2 = Array2::<f64>::zeros((n, m - 2));
    for j in 0..m - 2 {
        d2.column_mut(j).assign(&(d1.column(j + 1).to_owned() - d1.column(j)));
    }

    let first_diff = d1.column(0).to_owned();
    let d1_trunc   = d1.slice(s![.., ..m - 2]).to_owned(); // n × m-2

    // result = U[:,0] - D1_trunc @ (pinv(D2) @ first_diff)
    let correction = d1_trunc.dot(&pseudoinverse(&d2)?.dot(&first_diff));
    let result = iterates.column(0).to_owned() - correction;
    if !is_finite_array(&result) { return None; }
    Some(result)
}

// SEA / VEA — Epsilon extrapolation (shared table-building logic)
//
// Implements the Wynn epsilon algorithm on a table of iterates.
// Both methods alternate between two epsilon table rows using differences;
// SEA inverts element-wise, VEA inverts each column vector via v/‖v‖².
// Requires an odd number of iterates; drops one if even.
fn perform_epsilon(outputs: &[Array1<f64>], m: usize, sea: bool) -> Option<Array1<f64>> {
    let k = outputs.len();
    let n = outputs[0].len();

    // Epsilon algorithm is only defined for an odd column count
    let m_used = if m.is_multiple_of(2) { m - 1 } else { m };
    if m_used < 3 { return None; }

    // Build initial iterate matrix (n × m_used)
    let mut mat = Array2::<f64>::zeros((n, m_used));
    for (j, i) in (k - m_used..k).enumerate() {
        mat.column_mut(j).assign(&outputs[i]);
    }

    // `previous` holds the ε-table row from two steps back; starts at zero
    let mut previous = Array2::<f64>::zeros((n, m_used - 1));

    // Reduce: each pass shrinks mat by one column until one column remains
    for mc in (2..=m_used).rev() {
        // diff[:,j] = mat[:,j+1] - mat[:,j]  (n × mc-1)
        let diff = mat.slice(s![.., 1..mc]).to_owned()
                 - mat.slice(s![.., ..mc - 1]).to_owned();

        // Invert the differences
        let inv_diff = if sea || n == 1 {
            // SEA: element-wise reciprocal
            diff.mapv(|v| if v.abs() < 1e-300 { 0.0 } else { 1.0 / v })
        } else {
            // VEA: each column vector c → c / ‖c‖²  (Moore-Penrose pseudoinverse of a column)
            let mut result = Array2::<f64>::zeros(diff.dim());
            for j in 0..diff.ncols() {
                let col = diff.column(j);
                let norm_sq: f64 = col.iter().map(|&v| v * v).sum();
                if norm_sq > 1e-300 {
                    result.column_mut(j).assign(&(col.to_owned() / norm_sq));
                }
            }
            result
        };

        let new_mat = previous.slice(s![.., ..mc - 1]).to_owned() + inv_diff;
        previous = mat.slice(s![.., 1..mc - 1]).to_owned();
        mat = new_mat;
    }

    let result = mat.column(0).to_owned();
    if !is_finite_array(&result) { return None; }
    Some(result)
}
