use crate::{
    Dtype, SEED, data_structures::matrix::{Matrix, sum_cols}, layers::Layer
};

pub struct DenseLayer {
    weights: Matrix, // rows: output_size, cols: input_size
    biases: Matrix,  // rows: output_size, cols: 1

    velocity_w: Matrix,
    velocity_b: Matrix,

    input_cache: Matrix,
    weights_gradient: Matrix,
    biases_gradient: Matrix,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> DenseLayer {
        let weights = Matrix::new_seeded_random(output_size, input_size, SEED);
        let biases = Matrix::new_seeded_random(output_size, 1, SEED);

        let velocity_w = Matrix::new(output_size, input_size);
        let velocity_b = Matrix::new(output_size, 1);

        DenseLayer {
            weights,
            biases,

            velocity_w,
            velocity_b,

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

    /// Backward pass implements Gradient Descent with Momentum.
    fn backward(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: Dtype,
        momentum_factor: Dtype,
    ) -> Matrix {
        let batch_size = self.input_cache.cols as Dtype;

        let input_transposed = self.input_cache.transpose();
        let raw_weights_gradient = output_gradient * &input_transposed;

        let raw_biases_gradient = sum_cols(output_gradient);

        let weights_transposed = self.weights.transpose();
        let input_gradient = &weights_transposed * output_gradient;

        let scaled_learning_rate = learning_rate / batch_size;

        let current_gradient_w = &raw_weights_gradient * scaled_learning_rate;
        let current_gradient_b = &raw_biases_gradient * scaled_learning_rate;

        self.velocity_w = &(&self.velocity_w * momentum_factor) - &current_gradient_w;
        self.velocity_b = &(&self.velocity_b * momentum_factor) - &current_gradient_b;

        self.weights = &self.weights + &self.velocity_w;
        self.biases = &self.biases + &self.velocity_b;

        self.weights_gradient = raw_weights_gradient;
        self.biases_gradient = raw_biases_gradient;

        input_gradient
    }
}
