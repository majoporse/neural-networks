use crate::{Dtype, data_structures::matrix::Matrix, layers::Layer};

/// The Softmax activation layer (typically used as the output layer for classification).
pub struct Softmax {
    // Cache the output of the forward pass for use in the backward pass.
    output_cache: Matrix,
}

impl Softmax {
    pub fn new() -> Softmax {
        Softmax {
            output_cache: Matrix::new(0, 0),
        }
    }
}

impl Layer for Softmax {
    fn get_weights(&self) -> Option<&Matrix> {
        None
    }
    fn get_biases(&self) -> Option<&Matrix> {
        None
    }

    /// Forward pass: calculates Softmax(x) = exp(x) / sum(exp(x))
    /// input: Matrix of shape (features, batch_size)
    /// output: Matrix of shape (features, batch_size)
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut output = input.clone();

        for col in 0..input.cols {
            // Find max for numerical stability
            let mut max_val = Dtype::NEG_INFINITY;
            for row in 0..input.rows {
                max_val = max_val.max(input.get(row, col));
            }

            // Calculate exponentials and sum
            let mut sum_exp = 0.0;
            for row in 0..input.rows {
                let exp_val = (input.get(row, col) - max_val).exp();
                output.set(row, col, exp_val);
                sum_exp += exp_val;
            }

            // Normalize
            for row in 0..input.rows {
                let current_val = output.get(row, col);
                output.set(row, col, current_val / sum_exp);
            }
        }

        self.output_cache = output.clone();
        output
    }

    /// Backward pass for Softmax combined with Categorical Cross-Entropy Loss
    /// dL/dX = Y_pred - Y_true
    fn backward(&mut self, target_true: &Matrix, _learning_rate: Dtype, _momentum_factor: Dtype) -> Matrix {
        // The output_gradient is actually Y_true in this combined case
        // dL/dZ = Y_pred - Y_true
        &self.output_cache - target_true
    }
}
