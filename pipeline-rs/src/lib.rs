use base::{Pipeline, TextClassificationPipeline};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::prelude::*;
use onnxruntime::tensor;
use onnxruntime::{session::Session, LoggingLevel};
use std::cmp::Ordering;
use std::sync::Mutex;
use tokenizers::tokenizer::Tokenizer;

// use serde::{Deserialize, Serialize};

pub mod base;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
