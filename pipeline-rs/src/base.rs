use onnxruntime::environment::Environment;
use onnxruntime::{session::Session, LoggingLevel};
use tokenizers::tokenizer::Tokenizer;

pub trait Pipeline<Model, Processor, PreProcessingOutput, PredictionOutput, PostProcessingOutput> {
    fn new(model: &str, processor: &str) -> Self;
    // fn preprocess(&self, input: &str) -> PreProcessingOutput;
    // fn predict(&self, embeddings: PreProcessingOutput) -> PredictionOutput;
    // fn postprocess(&self, predictions: PredictionOutput) -> PostProcessingOutput;
}

pub struct TextClassificationPipeline {
    pub model: Session,
    pub processor: Tokenizer,
}

impl Pipeline<Environment, Tokenizer, String, String, String> for TextClassificationPipeline {
    fn new(model: &str, processor: &str) -> Self {
        let environment = Environment::builder()
            .with_name("app")
            .with_log_level(LoggingLevel::Verbose)
            .build()
            .unwrap();
        let session = environment
            .new_session_builder()
            .unwrap()
            .with_model_from_file(model)
            .unwrap();
        let tokenizer = Tokenizer::from_file(processor).unwrap();

        TextClassificationPipeline {
            model: session,
            processor: tokenizer,
        }
    }

    // fn preprocess(&self, input: &str) -> Vec<String> {
    //     self.processor.encode_list(input, 128)
    // }

    // fn predict(&self, embeddings: Vec<String>) -> Vec<f32> {
    //     let mut input = vec![];
    //     for embedding in embeddings {
    //         input.push(embedding);
    //     }
    //     let mut input_tensor = self.model.create_tensor(input);
    //     let mut output_tensor = self.model.create_tensor(vec![]);
    //     self.model
    //         .run(&[&mut input_tensor], &[&mut output_tensor], None)
    //         .unwrap();
    //     let output = output_tensor.to_vec();
    //     output
    // }

    // fn postprocess(&self, predictions: Vec<f32>) -> Vec<(String, f32)> {
    //     let mut result = vec![];
    //     for (i, prediction) in predictions.iter().enumerate() {
    //         result.push((i.to_string(), *prediction));
    //     }
    //     result
    // }
}
