use crate::{
    Dtype, callbacks::Callback, data_structures::matrix::Matrix, networks::network::Network,
};

pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: Dtype, // minimum improvement to reset counter
    best_loss: Dtype,
    wait: usize,
    pub stopped_epoch: usize,
    x_valid: Matrix,
    y_valid: Matrix,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: Dtype, x_valid: &Matrix, y_valid: &Matrix) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            best_loss: Dtype::INFINITY,
            wait: 0,
            stopped_epoch: 0,
            x_valid: x_valid.clone(),
            y_valid: y_valid.clone(),
        }
    }
}

impl Callback for EarlyStopping {
    fn on_train_end(&mut self, _network: &mut Network) {}

    fn on_epoch_end(&mut self, net: &mut Network, _y_pred: &Matrix, _y_true: &Matrix) -> bool {
        let (val_loss, _val_accuracy) = net.validate(&self.x_valid, &self.y_valid); // or pass validation set

        if val_loss + self.min_delta < self.best_loss {
            self.best_loss = val_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped_epoch += 1;
                log::info!(
                    "Early stopping triggered at epoch {}. Best validation loss: {:.6}",
                    self.stopped_epoch,
                    self.best_loss
                );
                return true;
            }
        }

        return false;
    }

    fn on_train_begin(&mut self) {
        self.best_loss = Dtype::INFINITY;
        self.wait = 0;
    }
}
