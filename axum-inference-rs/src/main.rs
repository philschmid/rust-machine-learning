use axum::{http::Method, response::IntoResponse, routing::post, AddExtensionLayer, Json, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use tower::ServiceBuilder;
use tower_http::{
    cors::{any, CorsLayer},
    trace::TraceLayer,
};

#[derive(Deserialize, Debug)]
pub struct PredictRequest {
    inputs: String,
}

#[derive(Serialize, Debug)]
struct PredictResponse {
    value: f32,
}

use pyo3::prelude::*;
use std::time::{Duration, Instant};

fn load_python_module_from_file() {
    let now = Instant::now();
    const PYTHON_MODULE: &str = include_str!("../python/handler.py");
    println!("Loading python as string {}", now.elapsed().as_millis());

    Python::with_gil(|py| {
        println!("Loading python with gil {}", now.elapsed().as_millis());
        let pipeline_file =
            PyModule::from_code(py, PYTHON_MODULE, "handler.py", "pipeline").unwrap();
        println!("Loading python module {}", now.elapsed().as_millis());
        let handler = pipeline_file.getattr("Handler").unwrap();
        println!("Loading handler module {}", now.elapsed().as_millis());
        let handler = handler.call1(("text-classification",)).unwrap();
        println!("Loading init handler {}", now.elapsed().as_millis());

        let pred = handler.call_method1("__call__", ("i love you.",)).unwrap();
        println!("predict {}", now.elapsed().as_millis());
        println!("{}", pred)
    })
}

async fn predict_route() -> impl IntoResponse {
    Json(PredictResponse { value: 2.0 })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set the RUST_LOG, if it hasn't been explicitly defined
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "axum_tokio_mpsc=debug,tower_http=debug")
    }
    // init request tracing for `TraceLayer`
    tracing_subscriber::fmt::init();

    load_python_module_from_file();

    // // middleware stack
    // let middleware_stack = ServiceBuilder::new()
    //     .layer(TraceLayer::new_for_http())
    //     .layer(
    //         // see https://docs.rs/tower-http/latest/tower_http/cors/index.html
    //         // for more details
    //         CorsLayer::new()
    //             .allow_origin(any())
    //             .allow_methods(vec![Method::POST, Method::OPTIONS])
    //             .allow_headers(any()),
    //     )
    //     .layer(AddExtensionLayer::new(tx.clone()));

    // // build our application with a route
    // let app = Router::new()
    //     .route("/predict", post(predict_route))
    //     .layer(middleware_stack);

    // // run it with hyper on localhost:3000
    // let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    // tracing::debug!("listening on {}", addr);
    // axum::Server::bind(&addr)
    //     .http1_preserve_header_case(true)
    //     .http1_title_case_headers(true)
    //     .serve(app.into_make_service())
    //     .await
    //     .unwrap();
    Ok(())
}
