


pub mod data_structures;
pub mod layers;
pub mod networks;
pub mod training;
pub mod callbacks;

type Dtype = f64;

fn main() {
    unsafe { std::env::set_var("RUST_LOG", "info") };

    env_logger::builder().format_source_path(true).init();

    log::info!("Starting XOR training...");
    training::xor::train_xor().unwrap();
}
