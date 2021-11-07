use tch::{Device, Tensor};
use tokenizers::{
    tokenizer::{Result, Tokenizer},
    DecoderWrapper, EncodeInput, Encoding, ModelWrapper, NormalizerWrapper, PaddingParams,
    PostProcessorWrapper, PreTokenizerWrapper, TokenizerImpl,
};

struct TorchscriptTokenizer {
    tokenizer: TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl TorchscriptTokenizer {
    pub fn new(identifier: &str) -> Self {
        let mut tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let test = tokenizer
            .with_padding(Some(PaddingParams {
                pad_to_multiple_of: Some(8u32 as usize),
                ..PaddingParams::default()
            }))
            .to_owned();
        TorchscriptTokenizer { tokenizer: test }
    }
}

trait MaxLengthTokenization {
    fn encode_with_max_length<'s, T>(
        &self,
        input: &str,
        max_length: u8,
        // truncate_sequences: bool,
    ) -> Result<Encoding>
    where
        T: Into<EncodeInput<'s>>;
}

impl MaxLengthTokenization
    for TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >
{
    fn encode_with_max_length<'s, T>(
        &self,
        input: &str,
        max_length: u8,
        // truncate_sequences: bool,
    ) -> Result<Encoding>
    where
        T: Into<EncodeInput<'s>>,
    {
        let mut temp_encodings = self.encode(input, true)?;
        temp_encodings.pad(
            max_length as usize,
            self.get_padding().unwrap().pad_id,
            self.get_padding().unwrap().pad_type_id,
            self.get_padding().unwrap().pad_token.as_str(),
            tokenizers::PaddingDirection::Right,
        );
        Ok(temp_encodings)
    }
}
