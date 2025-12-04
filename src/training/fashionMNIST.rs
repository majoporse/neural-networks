use crate::{
    Dtype, callbacks::{early_stopping::EarlyStopping, plotting_callback::PlottingCallback}, layers::{dense::{ConfigDenseLayer, DenseLayer}, relu::ReLULayer, softmax::Softmax}, networks::network::Network, testing::test_net::test_network, training::data_load::load_data
};

// --- MNIST Training Function (Placeholder Architecture) ---
pub fn train_mnist() -> anyhow::Result<()> {
    log::info!("\n==============================================");
    log::info!("--- FASHION MNIST Neural Network Training ---");
    log::info!("==============================================");

    // --- 1. Define Hyperparameters and Architecture (Minimalist) ---
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const H_SIZE: usize = 128;
    const LEARNING_RATE: Dtype = 0.007;
    const EPOCHS: usize = 10;
    const BATCH_SIZE: usize = 64;
    const MOMENTUM_FACTOR: Dtype = 0.01;
    const WEIGHT_DECAY: Dtype = 0.0001;
    const VALIDATION_SPLIT: f32 = 0.1;

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

    x_train = (&x_train - 128.0 as Dtype) / 128.0; // Normalize to [-1, 1]
    x_valid = (&x_valid - 128.0 as Dtype) / 128.0; // Normalize to [-1, 1]

    log::info!("dataset size: {}, {}", x_train.rows, x_train.cols);
    log::info!("validation size: {}, {}", x_valid.rows, x_valid.cols);

    let config = ConfigDenseLayer {
        learning_rate: LEARNING_RATE,
        momentum_factor: MOMENTUM_FACTOR,
        weight_decay: WEIGHT_DECAY,
    };

    let mut net = Network::new();
    net.add_layer(DenseLayer::new(INPUT_SIZE, H_SIZE, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(H_SIZE, 64, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(64, 32, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(32, OUTPUT_SIZE, &config));
    net.add_layer(Softmax::new());

    net.add_callback(PlottingCallback::new("./fashion_minst.png"));
    net.add_callback(EarlyStopping::new(5, 0.001, &x_valid, &y_valid));
    // net.add_callback(DebugCallback::new());

    log::info!("\n--- Starting Training for {} Epochs ---", EPOCHS);

    net.train(
        &x_train,
        &y_train,
        LEARNING_RATE,
        MOMENTUM_FACTOR,
        EPOCHS,
        BATCH_SIZE,
        WEIGHT_DECAY,
    )?;

    let final_pred = net.forward(&x_valid.split_into_batches(BATCH_SIZE)[0]);
    log::info!("\nFinal Predictions (Should be close to targets):");

    // viusalize the last batch
    for col in 0..final_pred.cols {
        let selected = (0..final_pred.rows)
            .into_iter()
            .find(|i| final_pred.get(*i, col) > 0.5);
        if selected.is_none() {
            continue;
        }
        let selected = selected.unwrap();
        let truth = (0..y_valid.rows)
            .into_iter()
            .find(|i| y_valid.get(*i, col) > 0.5);
        log::info!("selected: {} truth: {}", selected, truth.unwrap());
    }

    test_network(&mut net, INPUT_SIZE, OUTPUT_SIZE)?;

    Ok(())
}
