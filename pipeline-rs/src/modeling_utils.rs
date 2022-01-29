use onnxruntime::GraphOptimizationLevel;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use onnxruntime::environment::Environment;

use onnxruntime::{session::Session, LoggingLevel};

pub struct OnnxModel {
    pub session: Session,
    pub config: Config,
}

impl OnnxModel {
    pub fn from_file(model_path: &Path) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("app")
            .with_log_level(LoggingLevel::Verbose)
            .build()?;
        let session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            // .with_number_threads(4)?
            .with_model_from_file(model_path.join("model.onnx"))?;

        let config = Config::from_file(model_path)?;
        Ok(OnnxModel { session, config })
    }
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub id2label: HashMap<usize, String>,
}

impl Config {
    pub fn from_file(config_path: &Path) -> Result<Self> {
        // Open the file in read-only mode with buffer.
        let file = File::open(config_path.join("config.json"))?;
        let reader = BufReader::new(file);

        // Read the JSON contents of the file as an instance of `User`.
        let u: Config = serde_json::from_reader(reader)?;
        Ok(u)
    }
}
