
use crate::{
    Dtype,
    networks::network::Network,
    training::data_load::load_data,
};

// --- MNIST Training Function (Placeholder Architecture) ---
pub fn test_network(net: &mut Network, input: usize, output: usize) -> anyhow::Result<(Dtype, Dtype)> {
    log::info!("\n==============================================");
    log::info!("--- TEST Neural Network Training ---");
    log::info!("==============================================");

    // --- 2. Load FASHION MNIST Data ---
    let path_inputs =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_test_vectors.csv")?;
    let path_labels =
        std::fs::canonicalize("/home/xhatalc/pv021_project/data/fashion_mnist_test_labels.csv")?;

    let (mut x_train, y_train, x_valid, y_valid) = match load_data(
        path_inputs.to_str().unwrap(),
        path_labels.to_str().unwrap(),
        input,
        output,
        0.0,
    ) {
        Ok(data) => data,
        Err(e) => {
            log::error!("Error loading FASHION MNIST data: {:?}", e);
            return Err(e);
        }
    };
    x_train = &x_train / 255.0; // Normalize to [0, 1]

    log::info!("dataset size: {}, {}", x_train.rows, x_train.cols);
    log::info!("validation size: {}, {}", x_valid.rows, x_valid.cols);

    let (loss, acc) = net.validate(&x_train, &y_train);
    log::info!("Testing: Initial Loss: {:.6}, Accuracy: {:.2}%", loss, acc * 100.0);

    Ok((loss, acc))
}
