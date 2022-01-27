use axum::{
    extract::Extension, http::Method, response::IntoResponse, routing::post, AddExtensionLayer,
    Json, Router,
};
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

#[derive(FromPyObject, Debug, Serialize)]
struct TextClassificationResponse {
    #[pyo3(item)]
    label: String,
    #[pyo3(item)]
    score: f32,
}

use pyo3::{prelude::*, types::IntoPyDict};
use std::time::{Duration, Instant};

// fn load_python_module_from_file() {
//     let now = Instant::now();
//     println!("Loading python as string {}", now.elapsed().as_millis());

//     let handler = Python::with_gil(|py| {
//         println!("Loading python with gil {}", now.elapsed().as_millis());
//         let pipeline_file =
//             PyModule::from_code(py, PYTHON_MODULE, "handler.py", "Handler").unwrap();
//         println!("Loading python module {}", now.elapsed().as_millis());
//         let handler = pipeline_file
//             .getattr("Handler")
//             .unwrap()
//             .call1(("text-classification",))
//             .unwrap();

//         let pred = handler.call_method1("__call__", ("i love you.",)).unwrap();
//         println!("predict {}", now.elapsed().as_millis());
//         println!("{}", pred)
//     });
// }

async fn predict_route(
    Json(payload): Json<PredictRequest>,
    Extension(handler): Extension<Py<PyAny>>,
) -> impl IntoResponse {
    // println!("{:?}", payload);
    // let now = std::time::Instant::now();

    let pred = Python::with_gil(|py| {
        // println!("loading python {}", now.elapsed().as_millis());
        // let now = std::time::Instant::now();
        let res = handler
            .call1(py, (payload.inputs.as_str(),))
            .unwrap()
            .extract::<Vec<TextClassificationResponse>>(py)
            .unwrap();
        // println!("prediction took: {}ms", now.elapsed().as_millis());
        res
    });
    // println!("{:?}", pred);
    Json(pred)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set the RUST_LOG, if it hasn't been explicitly defined
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "axum_tokio_mpsc=debug,tower_http=debug")
    }
    // init request tracing for `TraceLayer`
    tracing_subscriber::fmt::init();

    let text_classification: Py<PyAny> = Python::with_gil(|py| {
        let kwargs = [("device", 0)].into_py_dict(py);
        const PYTHON_MODULE: &str = include_str!("../python/handler.py");

        let pipeline_file = PyModule::from_code(py, PYTHON_MODULE, "handler.py", "Handler")
            .expect("Handler should be loaded");
        pipeline_file
            .getattr("Handler")
            .unwrap()
            .call(("text-classification",), Some(kwargs))
            .unwrap()
            .extract()
            .unwrap()
    });

    // middleware stack
    let middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(
            // see https://docs.rs/tower-http/latest/tower_http/cors/index.html
            // for more details
            CorsLayer::new()
                .allow_origin(any())
                .allow_methods(vec![Method::POST, Method::OPTIONS])
                .allow_headers(any()),
        )
        .layer(AddExtensionLayer::new(text_classification.clone()));

    // build our application with a route
    let app = Router::new()
        .route("/predict", post(predict_route))
        .layer(middleware_stack);

    // run it with hyper on localhost:3000
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .http1_preserve_header_case(true)
        .http1_title_case_headers(true)
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}
