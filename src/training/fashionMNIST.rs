use std::collections::VecDeque;

use crate::{Dtype, callbacks::{debug_callback::DebugCallback, plotting_callback::PlottingCallback}, layers::{dense::DenseLayer, relu::ReLULayer, softmax::Softmax}, networks::network::Network, training::data_load::load_data};

// --- MNIST Training Function (Placeholder Architecture) ---
pub fn train_mnist() -> anyhow::Result<()> {
    log::info!("\n==============================================");
    log::info!("--- FASHION MNIST Neural Network Training ---");
    log::info!("==============================================");

    // --- 1. Define Hyperparameters and Architecture (Minimalist) ---
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const H_SIZE: usize = 128;
    const LEARNING_RATE: Dtype = 0.01;
    const EPOCHS: usize = 1;
    const BATCH_SIZE: usize = 256;
    // todo momentum
    // todo learning rate schedule

    // --- 2. Load FASHION MNIST Data ---
    let path_inputs = std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_vectors.csv")?;
    let path_labels = std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_labels.csv")?;

    let (input_x, y_true) = match load_data(
        path_inputs.to_str().unwrap(),
        path_labels.to_str().unwrap(),
        INPUT_SIZE,
        OUTPUT_SIZE,
        BATCH_SIZE,
    ) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Error loading FASHION MNIST data: {:?}", e);
            return Err(e);
        }
    };

    // --- 3. Assemble the Network (784 -> 128 -> 128 -> 10) ---
    let mut net = Network::new();
    net.add_layer(DenseLayer::new(INPUT_SIZE, H_SIZE));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, OUTPUT_SIZE));
    net.add_layer(Softmax::new());
    
    net.add_callback(PlottingCallback::new("./fashion_minst.png"));
    net.add_callback(DebugCallback::new());

    log::info!("\n--- Starting Training for {} Epochs ---", EPOCHS);

    net.train(&input_x, &y_true, LEARNING_RATE, EPOCHS)?;


    let final_pred = net.forward(&input_x[0]);
    log::info!("\nFinal Predictions (Should be close to targets):");

    for c in 0..final_pred.cols {
        let input_a = input_x[0].get(0, c);
        let input_b = input_x[0].get(1, c);
        let prob_false = final_pred.get(0, c);
        let prob_true = final_pred.get(1, c);
        log::info!(
            "   Input ({}, {}): [False: {:.4}, True: {:.4}]",
            input_a,
            input_b,
            prob_false,
            prob_true
        );
    }

    Ok(())
}
