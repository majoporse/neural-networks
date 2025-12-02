use crate::{data_structures::matrix::Matrix, networks::network::{Network, TrainingMetric}};

pub mod plotting_callback;
pub mod debug_callback;

pub trait Callback: Send {
    /// Called at the start of training.
    fn on_train_begin(&mut self) {}

    /// Called at the end of every epoch. The callback is responsible for calculating metrics.
    fn on_epoch_end(&mut self, network: &mut Network, y_pred: &Matrix, y_true: &Matrix);

    /// Called at the end of training. The callback can use the network for final analysis.
    fn on_train_end(&mut self, network: &mut Network);
}
