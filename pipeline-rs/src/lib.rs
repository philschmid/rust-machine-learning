mod base;


fn call() {
    let environment = Environment::builder()
    .with_name("app")
    .with_log_level(LoggingLevel::Verbose)
    .build()
    .unwrap();

let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();

let state =  environment
            .new_session_builder()
            .unwrap()
            .with_model_from_file("./pt_model.onnx")
            .unwrap(),
    
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
