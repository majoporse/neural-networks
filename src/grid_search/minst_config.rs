use crate::{
    Dtype, grid_search::train_config::TrainConfig, layers::{dense::DenseLayer, relu::ReLULayer, softmax::Softmax}, networks::network::Network, testing::test_net::test_network, training::data_load::load_data
};

pub fn train_mnist_with_config(config: &TrainConfig) -> anyhow::Result<Dtype> {
    log::info!("===== Running Config =====");
    log::info!("{:?}", config);
    log::info!("==========================");

    // --- Fixed network I/O sizes ---
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;

    // --- Extract config hyperparameters ---
    let h1 = config.hidden_size;
    let h2 = config.hidden_size_2;
    let h3 = config.hidden_size_3;
    let epochs = config.epochs;
    let lr = config.learning_rate;
    let bs = config.batch_size;
    let momentum = config.momentum;
    let weight_decay = config.weight_decay;
    let validation_split = 0.2;

    // --- Load data ---
    let path_inputs =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_vectors.csv")?;
    let path_labels =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_labels.csv")?;

    let (mut x_train, y_train, mut x_valid, y_valid) = load_data(
        path_inputs.to_str().unwrap(),
        path_labels.to_str().unwrap(),
        INPUT_SIZE,
        OUTPUT_SIZE,
        validation_split,
    )?;

    x_train = &x_train / 255.0; // Normalize to [0, 1]
    x_valid = &x_valid / 255.0; // Normalize to [0, 1]

    log::info!("Dataset size: {} samples", x_train.cols);

    // --- Build network as defined by config ---
    let config = crate::layers::dense::ConfigDenseLayer {
        learning_rate: lr,
        momentum_factor: momentum,
        weight_decay,
    };

    let mut net = Network::new();
    net.add_layer(DenseLayer::new(INPUT_SIZE, h1, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(h1, h2, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(h2, h3, &config));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(h3, OUTPUT_SIZE, &config));
    net.add_layer(Softmax::new());

    // For grid search, disable plots.
    // Enable manually when doing single-run training.
    // net.add_callback(PlottingCallback::new("./fashion_minst.png"));

    log::info!("Training for {} epochs…", epochs);

    net.train(&x_train, &y_train, epochs, bs)?;

    // --- Final evaluation on the full dataset ---
    let (_final_loss, accuracy) = test_network(&mut net, INPUT_SIZE, OUTPUT_SIZE)?;

    log::info!("Final accuracy: {:.4}", accuracy);

    Ok(accuracy)
}
