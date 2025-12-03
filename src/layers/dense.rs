use crate::{
    Dtype, SEED,
    data_structures::matrix::{Matrix, sum_cols},
    layers::Layer,
};

pub struct DenseLayer {
    weights: Matrix, // rows: output_size, cols: input_size
    biases: Matrix,  // rows: output_size, cols: 1

    velocity_w: Matrix,
    velocity_b: Matrix,

    input_cache: Matrix,
}

impl DenseLayer {
    // Modify the new constructor to accept weight_decay
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

    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input_cache = input.clone();

        let batch_size = input.cols;
        let mut output = &self.weights * input;

        for col in 0..batch_size {
            for row in 0..self.weights.rows {
                let current_val = output.get(row, col);
                output.set(row, col, current_val + self.biases.get(row, 0));
            }
        }
        output
    }

    fn backward(
        &mut self,
        output_gradient: &Matrix,
        learning_rate: Dtype,
        momentum_factor: Dtype,
        weight_decay: Dtype,
    ) -> Matrix {
        let batch_size = self.input_cache.cols as Dtype;

        // 1. Calculate Gradients regarding Input (unchanged)
        let weights_transposed = self.weights.transpose();
        let input_gradient = &weights_transposed * output_gradient;

        // 2. Calculate Standard Gradients
        let input_transposed = self.input_cache.transpose();
        let raw_weights_gradient = output_gradient * &input_transposed;
        let mut current_gradient_w = &raw_weights_gradient * (1.0 / batch_size); // Note: made mutable

        let raw_biases_gradient = sum_cols(output_gradient);
        let current_gradient_b = &raw_biases_gradient * (1.0 / batch_size);

        // ==========================================================
        // NEW: Apply Weight Decay (L2 Regularization Gradient)
        // L2 Grad = lambda * W
        // We add this to the gradient calculated from the loss function.
        // ==========================================================
        if weight_decay > 0.0 {
            let l2_grad_w = &self.weights * weight_decay;
            current_gradient_w = current_gradient_w + l2_grad_w;
            // Bias gradient (current_gradient_b) is typically left unchanged.
        }

        // 3. Update Velocities (Momentum Logic) (unchanged)
        // v = (v * momentum) + (gradient * learning_rate)
        self.velocity_w =
            (&self.velocity_w * momentum_factor) + (&current_gradient_w * learning_rate);
        self.velocity_b =
            (&self.velocity_b * momentum_factor) + (&current_gradient_b * learning_rate);

        // 4. Update Weights and Biases using the Velocity (unchanged)
        // w = w - v
        self.weights = &self.weights - &self.velocity_w;
        self.biases = &self.biases - &self.velocity_b;

        input_gradient
    }
}
