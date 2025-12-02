use crate::{Dtype, data_structures::matrix::Matrix};

pub mod dense;
pub mod softmax;
pub mod relu;


pub trait Layer {
    fn get_weights(&self) -> Option<&Matrix>;
    fn get_biases(&self) -> Option<&Matrix>;
    fn forward(&mut self, input: &Matrix) -> Matrix;

    fn backward(&mut self, output_gradient: &Matrix, learning_rate: Dtype) -> Matrix;
}