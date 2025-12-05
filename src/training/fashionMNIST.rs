use crate::{
    Dtype,
    callbacks::{early_stopping::EarlyStopping, plotting_callback::PlottingCallback},
    layers::{
        dense::{ConfigDenseLayer, DenseLayer},
        relu::ReLULayer,
        softmax::Softmax,
    },
    networks::network::Network,
    testing::test_net::test_network,
    training::data_load::load_data,
};

// --- MNIST Training Function (Placeholder Architecture) ---
pub fn train_mnist() -> anyhow::Result<()> {
    log::info!("\n==============================================");
    log::info!("--- FASHION MNIST Neural Network Training ---");
    log::info!("==============================================");

    // --- 1. Define Hyperparameters and Architecture (Minimalist) ---
    // --- 1. Define Hyperparameters and Architecture ---
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;

    const H1_SIZE: usize = 64;
    const H2_SIZE: usize = 32;
    const H3_SIZE: usize = 32;

    // Optimized Learning Rate for strong Momentum
    const LEARNING_RATE: Dtype = 0.001;
    const EPOCHS: usize = 30;
    const BATCH_SIZE: usize = 128;
    const MOMENTUM_FACTOR: Dtype = 0.1;
    const WEIGHT_DECAY: Dtype = 0.00001;
    const VALIDATION_SPLIT: f32 = 0.2;

    // --- 2. Load FASHION MNIST Data ---
    let path_inputs =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_vectors.csv")?;
    let path_labels =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_labels.csv")?;

    let (mut x_train, y_train, mut x_valid, y_valid) = match load_data(
        path_inputs.to_str().unwrap(),
        path_labels.to_str().unwrap(),
        INPUT_SIZE,
        OUTPUT_SIZE,
        VALIDATION_SPLIT,
    ) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Error loading FASHION MNIST data: {:?}", e);
            return Err(e);
        }
    };

    x_train = &x_train / 255.0; // Normalize to [0, 1]
    x_valid = &x_valid / 255.0; // Normalize to [0, 1]

    log::info!("dataset size: {}, {}", x_train.rows, x_train.cols);
    log::info!("validation size: {}, {}", x_valid.rows, x_valid.cols);

    let config = ConfigDenseLayer {
        learning_rate: LEARNING_RATE,
        momentum_factor: MOMENTUM_FACTOR,
        weight_decay: WEIGHT_DECAY,
    };

    let mut net = Network::new();
    net.add_layer(DenseLayer::new(INPUT_SIZE, H1_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H1_SIZE, H2_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H2_SIZE, H3_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H3_SIZE, OUTPUT_SIZE, &config));
    net.add_layer(Softmax::new());

    net.add_callback(PlottingCallback::new("fashion_minst"));
    // net.add_callback(EarlyStopping::new(10, 0.001, &x_valid, &y_valid));
    // net.add_callback(DebugCallback::new());

    log::info!("\n--- Starting Training for {} Epochs ---", EPOCHS);

    net.train(&x_train, &y_train, EPOCHS, BATCH_SIZE)?;

    let final_pred = net.forward(&x_valid.split_into_batches(BATCH_SIZE)[0]);
    log::info!("\nFinal Predictions (Should be close to targets):");

    // viusalize the last batch
    for col in 0..final_pred.cols {
        for row in 0..final_pred.rows {
            print!("{:.0} ", final_pred.get(row, col));
        }
        println!();
        for row in 0..y_valid.rows {
            print!("{:.0} ", y_valid.get(row, col));
        }
        println!("\n");
    }

    test_network(&mut net, INPUT_SIZE, OUTPUT_SIZE)?;

    Ok(())
}
