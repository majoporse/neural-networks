use crate::data_structures::matrix::Matrix;

pub mod adagrad;
pub mod adam;

pub trait Optimizer {
    fn update(&mut self, weights: &Matrix, biases: &Matrix, gradients: Matrix, bias_gradients: Matrix) -> (Matrix, Matrix);
}