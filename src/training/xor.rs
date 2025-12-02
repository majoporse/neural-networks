// Import ProgressIterator and ProgressStyle

// --- Removed all ratatui/crossterm imports ---
// use ratatui::prelude::*;
// use crossterm::{terminal::..., execute};
// use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

// Assuming these are defined in your project:
use crate::Dtype;
use crate::callbacks::debug_callback::DebugCallback;
use crate::callbacks::plotting_callback::PlottingCallback;
use crate::layers::dense::DenseLayer;
use crate::layers::relu::ReLULayer;
use crate::layers::softmax::Softmax;
use crate::networks::network::Network;
use crate::training::data_load::load_data;

// NOTE: You must have the calculate_loss and calculate_accuracy methods
// implemented on your Network struct for this to compile.

pub fn train_xor() -> anyhow::Result<()> {
    log::info!("\n==============================================");
    log::info!("--- XOR Neural Network Training ---");
    log::info!("==============================================");

    // --- 1. Define Hyperparameters and Architecture (Minimalist) ---
    const INPUT_SIZE: usize = 4;
    const OUTPUT_SIZE: usize = 2;
    const H_SIZE: usize = 5;
    const LEARNING_RATE: Dtype = 0.0001;
    const EPOCHS: usize = 5000;
    const BATCH_SIZE: usize = 5;
    const MOMENTUM_FACTOR: Dtype = 0.9;
    // todo momentum
    // todo learning rate schedule

    // --- 2. Load XOR Data ---
    let path_inputs = std::fs::canonicalize("../../../data/xor_4.csv")?;
    let path_labels = std::fs::canonicalize("../../../data/xor_4_labels.csv")?;

    let (input_x, y_true) = match load_data(
        path_inputs.to_str().unwrap(),
        path_labels.to_str().unwrap(),
        INPUT_SIZE,
        OUTPUT_SIZE,
        BATCH_SIZE,
    ) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Error loading XOR data: {:?}", e);
            return Err(e);
        }
    };
    // log::info!("data: {:?}", input_x);
    // log::info!("labels: {:?}", y_true);

    // --- 3. Assemble the Network (2 -> 4 -> 2) ---
    let mut net = Network::new();
    net.add_layer(DenseLayer::new(INPUT_SIZE, H_SIZE));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, OUTPUT_SIZE));
    net.add_layer(Softmax::new());

    net.add_callback(PlottingCallback::new("./training_plot.png"));
    net.add_callback(DebugCallback::new());

    log::info!("\n--- Starting Training for {} Epochs ---", EPOCHS);

    net.train(&input_x, &y_true, LEARNING_RATE,  MOMENTUM_FACTOR, EPOCHS)?;

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
