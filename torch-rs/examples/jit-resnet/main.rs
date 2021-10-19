use tch::nn::Module;
use tch::vision::imagenet::{load_image_and_resize224, CLASSES, CLASS_COUNT};
use tch::{self, Tensor};

pub fn top(tensor: &Tensor, k: i64) -> Vec<(f64, String)> {
    let tensor = match tensor.size().as_slice() {
        [CLASS_COUNT] => tensor.shallow_clone(),
        [1, CLASS_COUNT] => tensor.view((CLASS_COUNT,)),
        [1, 1, CLASS_COUNT] => tensor.view((CLASS_COUNT,)),
        _ => panic!("unexpected tensor shape {:?}", tensor),
    };
    let (values, indexes) = tensor.topk(k, 0, true, true);
    let values = Vec::<f64>::from(values);
    let indexes = Vec::<i64>::from(indexes);
    values
        .iter()
        .zip(indexes.iter())
        .map(|(&value, &index)| (value, CLASSES[index as usize].to_owned()))
        .collect()
}

fn main() {
    // Load the Python saved module.
    let model = tch::CModule::load("resnet.pt").unwrap();

    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = load_image_and_resize224("image.jpeg").unwrap();
    let input_tensor = image.unsqueeze(0);

    // Apply the forward pass of the model to get the logits.
    let output = model.forward(&input_tensor).softmax(-1, tch::Kind::Float);
    println!("{:?}", output.size());
    // Print the top 5 categories for this image.
    for (probability, class) in top(&output, 15).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
}
