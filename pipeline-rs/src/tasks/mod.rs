pub mod text_classification;

use std::path::Path;

use anyhow::Result;

use ndarray::OwnedRepr;
use onnxruntime::ndarray::{prelude::*, IxDynImpl};
use tokenizers::tokenizer::Tokenizer;

use crate::modeling_utils::OnnxModel;

pub type EmbeddingArray =
    Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>>;

pub trait Pipeline<Model, Processor, PostProcessingOutput> {
    fn new(model: OnnxModel, tokenizer: Tokenizer) -> Self;
    fn from_path(path: &Path) -> Result<Self>
    where
        Self: Sized;
    fn preprocess(&self, input: &str) -> Result<EmbeddingArray>;
    fn predict(
        &mut self,
        embeddings: EmbeddingArray,
    ) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>>;
    fn postprocess(
        &mut self,
        predictions: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    ) -> Result<PostProcessingOutput>;
    fn call(&mut self, input: &str) -> Result<PostProcessingOutput> {
        let embeddings = self.preprocess(input)?;
        let predictions = self.predict(embeddings)?;
        self.postprocess(predictions)
    }
}
