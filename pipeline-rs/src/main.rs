use std::path::Path;

use anyhow::{bail, Result};
use lib::{
    base::Pipeline, modeling_utils::OnnxModel,
    tasks::text_classification::TextClassificationPipeline,
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let path = Path::new("model");
    let model = OnnxModel::from_file(path)?;
    let tokenizer = match Tokenizer::from_file(path.join("tokenizer.json")) {
        Ok(tk) => tk,
        Err(err) => bail!("{}", err),
    };

    let mut pipeline = TextClassificationPipeline::new(model, tokenizer);

    // let input = "I like you. I love you.";
    let input = "I hate you.";
    let res = pipeline.call(input)?;
    println!("{:?}", res);
    Ok(())
}
