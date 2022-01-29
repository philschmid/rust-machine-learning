use anyhow::Result;
use ndarray_stats::QuantileExt;
use std::ops::Deref;

use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use onnxruntime::tensor;
use tokenizers::Tokenizer;

use crate::{
    base::{EmbeddingArray, Pipeline},
    modeling_utils::OnnxModel,
    statistical_utils::softmax,
};

pub struct TextClassificationPipeline {
    pub model: OnnxModel,
    pub tokenizer: Tokenizer,
}

#[derive(Debug, Clone)]
pub struct TextClassificationOutput {
    pub label: String,
    pub score: f32,
}

impl Pipeline<OnnxModel, Tokenizer, TextClassificationOutput> for TextClassificationPipeline {
    fn new(model: OnnxModel, tokenizer: Tokenizer) -> Self {
        TextClassificationPipeline { model, tokenizer }
    }

    fn preprocess(&self, input: &str) -> Result<EmbeddingArray> {
        let encoding = self.tokenizer.encode(input, true).unwrap();

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
        Ok(vec![id_array, mask_array])
    }
    fn predict(
        &mut self,
        embeddings: EmbeddingArray,
    ) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> {
        let _outputs: Vec<tensor::OrtOwnedTensor<f32, _>> = self.model.session.run(embeddings)?;

        Ok(_outputs[0].deref().to_owned())
    }
    fn postprocess(
        &mut self,
        predictions: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    ) -> Result<TextClassificationOutput> {
        let scores = softmax(predictions)?;

        let label = self
            .model
            .config
            .id2label
            .get(&scores.argmax()?.1)
            .unwrap_or(&format!("LABEL_OF_IDX_{}", &scores.argmax()?.1))
            .to_string();

        let result = TextClassificationOutput {
            label,
            score: scores.max()?.to_owned(),
        };
        Ok(result)
    }
}
