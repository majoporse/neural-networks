use crate::{callbacks::Callback, data_structures::matrix::Matrix, networks::network::Network};


#[derive(Default)]
pub struct DebugCallback{
}

impl DebugCallback {
    pub fn new() -> Self {
        DebugCallback{}
    }
}

impl Callback for DebugCallback {
    fn on_epoch_end(&mut self, network: &mut Network, y_pred: &Matrix, y_true: &Matrix) {
    }

    fn on_train_end(&mut self, network: &mut Network) {

        for layer in &network.layers {
            log::info!("weights: {:?}", layer.get_weights());
            log::info!("biases: {:?}", layer.get_biases());
        }
    }
}
