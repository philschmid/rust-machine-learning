use anyhow::{bail, Result};
use std::path::Path;

use tokenizers::Tokenizer;

pub trait TokenizerUtils {
    fn from_path(path: &Path) -> Result<Tokenizer> {
        match Tokenizer::from_file(path.join("tokenizer.json")) {
            Ok(tk) => Ok(tk),
            Err(err) => bail!("{}", err),
        }
    }
}

impl TokenizerUtils for Tokenizer {}
