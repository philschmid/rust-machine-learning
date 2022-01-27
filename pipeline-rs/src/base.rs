trait Pipeline<Model, Processor, PreProcessingOutput, PredictionOutput, PostProcessingOutput> {
    let model: Model;
    let processor: Processor;
    fn new(&self, model: Model, processor: Processor) -> Self;
    fn preprocess(&self, input: &str) -> PreProcessingOutput;
    fn predict(&self, embeddings: PreProcessingOutput) -> PredictionOutput;
    fn postprocess(&self, predictions: PredictionOutput) -> PostProcessingOutput;
}
