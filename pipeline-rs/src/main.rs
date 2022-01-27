use std::ops::Deref;

use lib::base::{Pipeline, TextClassificationPipeline};
use ndarray_stats::QuantileExt;
use onnxruntime::{
    ndarray::{prelude::*, IxDynImpl, OwnedRepr},
    tensor,
};

pub fn softmax(logits: ArrayD<f32>) -> Array2<f32> {
    let mut softmax = logits.to_owned().into_dimensionality::<Ix2>().unwrap();
    // Calculate softmax
    let max = softmax.fold_axis(Axis(1), 0.0, |x, y| if *x > *y { *x } else { *y });
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x = (*x - max[b]).exp();
    }
    let sum = softmax.sum_axis(Axis(1));
    for ((b, _), x) in softmax.indexed_iter_mut() {
        *x /= sum[b];
    }
    softmax
}

fn preprocess(
    pipeline: &TextClassificationPipeline,
    input: &str,
) -> Vec<ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>> {
    let encoding = pipeline.processor.encode(input, true).unwrap();

    let ids = encoding
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();
    let id_array = Array::from_shape_vec((1, ids.len()), ids).unwrap();

    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    let mask_array = Array::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap();
    vec![id_array, mask_array]
}
fn predict(
    mut pipeline: TextClassificationPipeline,
    input: Vec<ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    let _outputs: Vec<tensor::OrtOwnedTensor<f32, _>> = pipeline.model.run(input).unwrap();

    _outputs[0].deref().to_owned()
}
fn postprocess(logits: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) {
    let scores = softmax(logits);
    println!("score: {}", scores.max().unwrap());

    let label_idx = scores.argmax();
    println!("index: {}", label_idx.unwrap().1);
}

fn main() {
    let pipeline = TextClassificationPipeline::new("model/model.onnx", "model/tokenizer.json");

    let input = "I like you. I love you.";
    // let input = "I hate you.";
    let embeddings = preprocess(&pipeline, input);
    let logits = predict(pipeline, embeddings);
    postprocess(logits);
}
