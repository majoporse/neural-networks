use crate::{data_structures::matrix::Matrix, layers::Layer};

pub struct ReLULayer {
    // Cache the input (Z) from the forward pass for use in the backward pass.
    input_cache: Matrix,
}

impl ReLULayer {
    pub fn new() -> ReLULayer {
        ReLULayer {
            input_cache: Matrix::new(0, 0),
        }
    }
}

impl Layer for ReLULayer {
    fn get_weights(&self) -> Option<&Matrix> {
        None
    }
    fn get_biases(&self) -> Option<&Matrix> {
        None
    }

    /// Forward pass: applies max(0, x) element-wise.
    /// input: Matrix of shape (features, batch_size)
    /// output: Matrix of shape (features, batch_size)
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input_cache = input.clone(); // Cache input (Z)

        let mut output = input.clone();
        for val in output.data.iter_mut() {
            *val = if *val > 0.0 { *val } else { 0.00 * *val };
        }
        output
    }

    /// Backward pass: dL/dX = dL/dY * ReLU'(X)
    /// ReLU'(x) is 1 if x > 0, and 0 otherwise.
    fn backward(&mut self, output_gradient: &Matrix) -> Matrix {
        // The learning_rate is ignored as activation layers have no trainable parameters.

        let mut relu_derivative = self.input_cache.clone();

        // Calculate the ReLU derivative mask: 1.0 where input was > 0, 0.0 otherwise.
        for val in relu_derivative.data.iter_mut() {
            *val = if *val > 0.0 { 1.0 } else { 0.00 };
        }

        // Apply the chain rule: Hadamard product (element-wise multiplication)
        output_gradient.element_wise_mul(&relu_derivative)
    }
}
