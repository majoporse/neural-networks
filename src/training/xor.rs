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
    const H_SIZE: usize = 10;
    const LEARNING_RATE: Dtype = 0.01;
    const EPOCHS: usize = 50000;
    const BATCH_SIZE: usize = 15;
    const MOMENTUM_FACTOR: Dtype = 0.09;
    // todo momentum
    // todo learning rate schedule

    // --- 2. Load XOR Data ---
    let path_inputs = std::fs::canonicalize("../../../data/xor_4.csv")?;
    let path_labels = std::fs::canonicalize("../../../data/xor_4_labels.csv")?;

    let (input_x, y_true, x_valid, y_valid) = match load_data(
        path_inputs.to_str().unwrap(),
        path_labels.to_str().unwrap(),
        INPUT_SIZE,
        OUTPUT_SIZE,
        0.1,
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
    let config = crate::layers::dense::ConfigDenseLayer {
        learning_rate: LEARNING_RATE,
        momentum_factor: MOMENTUM_FACTOR,
        weight_decay: 0.0,
    };
    let mut net = Network::new();
    net.add_layer(DenseLayer::new(INPUT_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, OUTPUT_SIZE, &config));
    net.add_layer(Softmax::new());

    net.add_callback(PlottingCallback::new("./training_plot.png"));
    net.add_callback(DebugCallback::new());

    log::info!("\n--- Starting Training for {} Epochs ---", EPOCHS);

    net.train(
        &input_x,
        &y_true,
        LEARNING_RATE,
        MOMENTUM_FACTOR,
        EPOCHS,
        BATCH_SIZE,
        0.01,
    )?;

    let final_pred = net.forward(&input_x.split_into_batches(BATCH_SIZE)[0]);
    log::info!("\nFinal Predictions (Should be close to targets):");

    for col in 0..final_pred.cols {
        let selected = (0..final_pred.rows)
            .into_iter()
            .find(|i| final_pred.get(*i, col) > 0.5);
        if selected.is_none() {
            continue;
        }
        let selected = selected.unwrap();
        let truth = (0..y_true.rows)
            .into_iter()
            .find(|i| y_true.get(*i, col) > 0.5);
        log::info!("selected: {} truth: {}", selected, truth.unwrap());
    }

    Ok(())
}
