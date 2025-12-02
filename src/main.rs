use rayon::ThreadPoolBuilder;

pub mod callbacks;
pub mod data_structures;
pub mod layers;
pub mod networks;
pub mod training;

type Dtype = f32;

pub const SEED: u64 = 42;

pub fn initialize_rayon_pool(num_threads: usize) {
    // This should only be called once, before any parallel operation.
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("Rayon thread pool set to {} threads.", num_threads);
}

fn main() {
    unsafe { std::env::set_var("RUST_LOG", "info") };

    env_logger::builder().format_source_path(true).init();
    initialize_rayon_pool(16);
    // log::info!("Starting XOR training...");
    // training::xor::train_xor().unwrap();
    log::info!("Starting Fashion MNIST training...");
    training::fashionMNIST::train_mnist().unwrap();
}
