use std::collections::VecDeque;

use crate::{Dtype, layers::{dense::DenseLayer, relu::ReLULayer, softmax::Softmax}, networks::network::Network, training::data_load::load_data};

// --- MNIST Training Function (Placeholder Architecture) ---
fn train_mnist() {
    println!("\n==============================================");
    println!("--- Fashion-MNIST Training Skeleton ---");
    println!("==============================================");
    
    // --- 1. Define Architecture (Tapered) ---
    const INPUT_SIZE: usize = 784; // 28*28 pixels
    const OUTPUT_SIZE: usize = 10; // 10 classes
    const H1_SIZE: usize = 256;
    const H2_SIZE: usize = 128;
    const H3_SIZE: usize = 64;
    const BATCH_SIZE: usize = 64;
    const LEARNING_RATE: Dtype = 0.01;
    const EPOCHS: usize = 5; // Small number of epochs for skeleton test
    
    // --- 2. Load MNIST Data ---
    let (input_x, y_true) = match load_data(
        "../../data/fashion_mnist_inputs.csv",
        "../../data/fashion_mnist_labels.csv",
        INPUT_SIZE,
        OUTPUT_SIZE,
    ) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading MNIST data: {}", e);
            return;
        }
    };
    
    assert_eq!(input_x.rows, INPUT_SIZE, "Input size mismatch after parsing.");
    assert_eq!(y_true.rows, OUTPUT_SIZE, "Output size mismatch after parsing.");
    println!("MNIST Dataset mock loaded (Inputs: {}x{}, Targets: {}x{}).", INPUT_SIZE, BATCH_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    
    // --- 3. Assemble the Tapered Network ---
    let mut net = Network::new();

    // H1: Dense (784 -> 256)
    net.add_layer(DenseLayer::new(INPUT_SIZE, H1_SIZE));
    net.add_layer(ReLULayer::new());
    
    // H2: Dense (256 -> 128)
    net.add_layer(DenseLayer::new(H1_SIZE, H2_SIZE));
    net.add_layer(ReLULayer::new());
    
    // H3: Dense (128 -> 64)
    net.add_layer(DenseLayer::new(H2_SIZE, H3_SIZE));
    net.add_layer(ReLULayer::new());
    
    // Output: Dense (64 -> 10)
    net.add_layer(DenseLayer::new(H3_SIZE, OUTPUT_SIZE));
    
    // Output Activation
    net.add_layer(Softmax::new());
    
    println!("MNIST Tapered Architecture (784 -> 256 -> 128 -> 64 -> 10) assembled with ReLU.");
    
    // --- 4. Mock Training Loop ---
    let mut loss_history = VecDeque::new();

    for epoch in 1..=EPOCHS {
        // FORWARD PASS
        let y_pred = net.forward(&input_x);
        
        let loss = net.calculate_loss(&y_pred, &y_true);
        loss_history.push_back(loss);

        // BACKWARD PASS & UPDATE
        net.backward(&y_true, LEARNING_RATE);
        
        // Reporting
        println!("Epoch {}/{}: Mock Loss = {:.6}", epoch, EPOCHS, loss);
    }
    
    println!("\n--- MNIST Training Skeleton Complete ---");
    println!("Final Mock Loss: {:.6}", loss_history.back().copied().unwrap_or(0.0));
    println!("Run this function with actual data and ReLU layers to train.");
}
