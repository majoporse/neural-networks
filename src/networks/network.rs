use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::{Dtype, callbacks::Callback, data_structures::matrix::Matrix, layers::Layer};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct TrainingMetric {
    pub epoch: u32,
    pub loss: Dtype,
    pub accuracy: Dtype,
}
pub struct Network {
    pub(crate) layers: Vec<Box<dyn Layer>>,
    bar_style: ProgressStyle,
    callbacks: Vec<Box<dyn Callback>>,
}

impl Network {
    pub fn new() -> Network {
        Network {
            layers: Vec::new(),
            bar_style: ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap(),
            callbacks: Vec::new(),
        }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn add_callback<C: Callback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::<C>::new(callback));
    }

    /// Performs the forward pass through all layers.
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    /// Performs the backward pass (gradient descent).
    pub fn backward(&mut self, y_true: &Matrix, learning_rate: Dtype) {
        let last_index = self.layers.len() - 1;
        // Start backward pass using the combined Softmax + Loss derivative
        // Note: In a real implementation, the Softmax + Loss derivative would typically be done here
        let mut gradient = self.layers[last_index].backward(y_true, learning_rate);

        // Iterate backward through the rest of the layers
        for i in (0..last_index).rev() {
            gradient = self.layers[i].backward(&gradient, learning_rate);
        }
    }

    /// Calculates the Categorical Cross-Entropy Loss.
    pub fn calculate_loss(&self, y_pred: &Matrix, y_true: &Matrix) -> Dtype {
        let batch_size = y_pred.cols as Dtype;
        let num_classes = y_pred.rows;
        let mut loss = 0.0;

        for c in 0..y_pred.cols {
            for r in 0..num_classes {
                let p = y_pred.get(r, c).max(1e-15); // Clamp for stability
                let t = y_true.get(r, c);
                // Categorical Cross-Entropy L = - sum(t * log(p))
                loss += t * p.ln();
            }
        }

        // Return average negative log-likelihood
        -loss / batch_size
    }

    /// Calculates the classification accuracy.
    pub fn calculate_accuracy(&self, y_pred: &Matrix, y_true: &Matrix) -> Dtype {
        let batch_size = y_pred.cols;
        let num_classes = y_pred.rows;
        let mut correct_predictions = 0;

        for c in 0..batch_size {
            // Find the predicted class (max probability index)
            let mut max_pred_val = -1.0;
            let mut predicted_class = 0;
            for r in 0..num_classes {
                if y_pred.get(r, c) > max_pred_val {
                    max_pred_val = y_pred.get(r, c);
                    predicted_class = r;
                }
            }

            // Find the true class (one-hot encoded index)
            let mut true_class = 0;
            for r in 0..num_classes {
                if y_true.get(r, c) > 0.9 {
                    true_class = r;
                    break;
                }
            }

            if predicted_class == true_class {
                correct_predictions += 1;
            }
        }

        correct_predictions as Dtype / batch_size as Dtype
    }

    /// Training loop executes all registered callbacks.
    pub fn train(
        &mut self,
        input_x: &Matrix,
        y_true: &Matrix,
        learning_rate: Dtype,
        epochs: usize,
    ) -> anyhow::Result<()> {

        for callback in self.callbacks.iter_mut() {
            callback.on_train_begin();
        }

        let bar = ProgressBar::new(epochs as u64);
        bar.set_style(self.bar_style.clone());

        for epoch in 1..=epochs {

            let y_pred = self.forward(&input_x);

            self.backward(&y_true, learning_rate);

            bar.inc(1);

            if epoch % 10 == 0 || epoch == 1 {
                let mut callbacks_vec = std::mem::take(&mut self.callbacks);

                for callback in callbacks_vec.iter_mut() {
                    callback.on_epoch_end(self, &y_pred, &y_true);
                }

                self.callbacks = callbacks_vec;

                let loss = self.calculate_loss(&y_pred, y_true);
                let accuracy = self.calculate_accuracy(&y_pred, y_true);
                bar.set_message(format!("Loss: {:.6} | Acc: {:.4}", loss, accuracy));

                bar.tick();
            }
        }

        bar.finish_with_message(format!("Training Complete."));
        log::info!("Training finished successfully.");

        let mut callbacks_vec = std::mem::take(&mut self.callbacks);

        for callback in callbacks_vec.iter_mut() {
            callback.on_train_end(self);
        }

        self.callbacks = callbacks_vec;

        Ok(())
    }
}
