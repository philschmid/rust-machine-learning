use anyhow::Result;
use ndarray::{Array2, ArrayD, Axis, Ix2};

pub fn softmax(logits: ArrayD<f32>) -> Result<Array2<f32>> {
    let mut softmax = logits.to_owned().into_dimensionality::<Ix2>()?;
    // Calculate softmax
    let max = softmax.fold_axis(Axis(1), 0.0, |x, y| if *x > *y { *x } else { *y });
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x = (*x - max[b]).exp();
    }
    let sum = softmax.sum_axis(Axis(1));
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x /= sum[b];
    }
    Ok(softmax)
}
