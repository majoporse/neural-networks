use crate::{
    Dtype,
    callbacks::plotting_callback::PlottingCallback,
    layers::{dense::DenseLayer, relu::ReLULayer, softmax::Softmax},
    networks::network::Network,
    training::data_load::load_data,
};

// --- MNIST Training Function (Placeholder Architecture) ---
pub fn train_mnist() -> anyhow::Result<()> {
    log::info!("\n==============================================");
    log::info!("--- FASHION MNIST Neural Network Training ---");
    log::info!("==============================================");

    // --- 1. Define Hyperparameters and Architecture (Minimalist) ---
    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const H_SIZE: usize = 64;
    const LEARNING_RATE: Dtype = 0.000001;
    const EPOCHS: usize = 15;
    const BATCH_SIZE: usize = 64;
    const MOMENTUM_FACTOR: Dtype = 0.0009;
    // todo momentum
    // todo learning rate schedule

    // --- 2. Load FASHION MNIST Data ---
    let path_inputs =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_vectors.csv")?;
    let path_labels =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_train_labels.csv")?;

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
    net.add_layer(DenseLayer::new(H_SIZE, 20));
    net.add_layer(ReLULayer::new());
    net.add_layer(DenseLayer::new(20, OUTPUT_SIZE));
    net.add_layer(Softmax::new());

    net.add_callback(PlottingCallback::new("./fashion_minst.png"));
    // net.add_callback(DebugCallback::new());

    log::info!("\n--- Starting Training for {} Epochs ---", EPOCHS);

    net.train(&input_x, &y_true, LEARNING_RATE, MOMENTUM_FACTOR, EPOCHS)?;

    let final_pred = net.forward(&input_x[0]);
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
        let truth = (0..y_true[0].rows)
            .into_iter()
            .find(|i| y_true[0].get(*i, col) > 0.5);
        log::info!("selected: {} truth: {}", selected, truth.unwrap());
    }
    Ok(())
}
