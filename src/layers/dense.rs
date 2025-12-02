use crate::{
    Dtype,
    data_structures::matrix::{Matrix, sum_cols},
    layers::Layer,
};

pub struct DenseLayer {
    weights: Matrix, // rows: output_size, cols: input_size
    biases: Matrix,  // rows: output_size, cols: 1

    // Cached values for backpropagation
    input_cache: Matrix,
    weights_gradient: Matrix,
    biases_gradient: Matrix,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> DenseLayer {
        let weights = Matrix::new_random(output_size, input_size);

        let biases = Matrix::new_random(output_size, 1);

        DenseLayer {
            weights,
            biases,

            input_cache: Matrix::new(0, 0),
            weights_gradient: Matrix::new(0, 0),
            biases_gradient: Matrix::new(0, 0),
        }
    }
}

impl Layer for DenseLayer {
    fn get_weights(&self) -> Option<&Matrix> {
        Some(&self.weights)
    }
    fn get_biases(&self) -> Option<&Matrix> {
        Some(&self.biases)
    }
    /// Output = W * Input + Bias
    /// input: Matrix of shape (input_size, batch_size)
    /// output: Matrix of shape (output_size, batch_size)
    fn forward(&mut self, input: &Matrix) -> Matrix {
        // Cache input for backpropagation
        self.input_cache = input.clone();

        let batch_size = input.cols;
        let mut output = &self.weights * input;

        // Add bias vector to every column (sample) in the output matrix
        for col in 0..batch_size {
            for row in 0..self.weights.rows {
                let current_val = output.get(row, col);
                output.set(row, col, current_val + self.biases.get(row, 0));
            }
        }

        output
    }

    fn backward(&mut self, output_gradient: &Matrix, learning_rate: Dtype) -> Matrix {
        let batch_size = self.input_cache.cols as Dtype;

        // 1. Calculate gradient for Weights (dW): dW = dL/dY * X^T
        let input_transposed = self.input_cache.transpose();
        self.weights_gradient = output_gradient * &input_transposed;

        // 2. Calculate gradient for Biases (dB): dB = sum(dL/dY across all samples)
        self.biases_gradient = sum_cols(output_gradient);
        // log::info!("Biases gradient: {:?} {:?}", self.biases_gradient, output_gradient);
        // std::thread::sleep(std::time::Duration::from_millis(10000000));

        // 3. Calculate gradient for Input (dX): dX = W^T * dL/dY (passed to previous layer)
        let weights_transposed = self.weights.transpose();
        let input_gradient = &weights_transposed * output_gradient;

        // 4. Update Weights and Biases
        // Update W: W = W - LR * dW / batch_size
        let weight_update = &self.weights_gradient * (learning_rate / batch_size);
        self.weights = &self.weights - &weight_update;

        // Update B: B = B - LR * dB / batch_size
        let bias_update = &self.biases_gradient * (learning_rate / batch_size);
        self.biases = &self.biases - &bias_update;

        input_gradient
    }
}
