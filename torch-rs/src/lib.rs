use tch::Tensor;

pub fn test_tensor() -> Tensor {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    t * 2
}
