use crate::{
    Dtype, SEED,
    data_structures::matrix::{Matrix, sum_cols},
    layers::{
        Layer,
        optimizers::{Optimizer, adagrad::AdaGrad},
    },
};

pub struct DenseLayer {
    weights: Matrix, // rows: output_size, cols: input_size
    biases: Matrix,  // rows: output_size, cols: 1

    input_cache: Matrix,
    optimizer: Box<dyn Optimizer>,
}

pub struct ConfigDenseLayer {
    pub learning_rate: Dtype,
    pub momentum_factor: Dtype,
    pub weight_decay: Dtype,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        config: &ConfigDenseLayer,
    ) -> DenseLayer {
        let weights = Matrix::new_seeded_random(output_size, input_size, SEED);
        let biases = Matrix::new_seeded_random(output_size, 1, SEED);

        let optimizers = Box::new(AdaGrad::new(
            config.learning_rate,
            1e-8,
            config.momentum_factor,
            config.weight_decay,
            input_size,
            output_size,
        ));
        DenseLayer {
            weights,
            biases,
            input_cache: Matrix::new(0, 0),
            optimizer: optimizers,
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

    fn backward(&mut self, output_gradient: &Matrix) -> Matrix {
        let batch_size = self.input_cache.cols as Dtype;

        let weights_transposed = self.weights.transpose();
        let input_gradient = &weights_transposed * output_gradient;

        let input_transposed = self.input_cache.transpose();
        let raw_weights_gradient = output_gradient * &input_transposed;
        let current_gradient_w = &raw_weights_gradient * (1.0 / batch_size);

        let raw_biases_gradient = sum_cols(output_gradient);
        let current_gradient_b = &raw_biases_gradient * (1.0 / batch_size);

        (self.weights, self.biases) = self.optimizer.update(
            &self.weights,
            &self.biases,
            current_gradient_w,
            current_gradient_b,
        );

        input_gradient
    }
}
