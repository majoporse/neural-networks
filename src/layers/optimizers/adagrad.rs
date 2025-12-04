use crate::{Dtype, data_structures::matrix::Matrix, layers::optimizers::Optimizer};

pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,

    grad_accum_w: Matrix,
    grad_accum_b: Matrix,

    velocity_w: Matrix,
    velocity_b: Matrix,

    momentum_factor: Dtype,
    weight_decay: Dtype,
}

impl AdaGrad {
    pub fn new(
        learning_rate: f32,
        epsilon: f32,
        momentum_factor: Dtype,
        weight_decay: Dtype,
        input_size: usize,
        output_size: usize,
    ) -> AdaGrad {
        AdaGrad {
            learning_rate,
            epsilon,

            velocity_w: Matrix::new(output_size, input_size),
            velocity_b: Matrix::new(output_size, 1),

            grad_accum_w: Matrix::new(output_size, input_size),
            grad_accum_b: Matrix::new(output_size, 1),

            momentum_factor,
            weight_decay,
        }
    }
}

impl Optimizer for AdaGrad {
    fn update(
        &mut self,
        weights: &Matrix,
        biases: &Matrix,
        mut weights_gradients: Matrix,
        bias_gradients: Matrix,
    ) -> (Matrix, Matrix) {

        // 3. Weight decay (L2)
        if self.weight_decay > 0.0 {
            let l2_grad_w = &*weights * self.weight_decay;
            weights_gradients = &weights_gradients + &l2_grad_w;
        }

        // ----------------------------
        // 4. AdaGrad adjustment
        // ----------------------------
        let grad_w_sq = weights_gradients.element_wise_mul(&weights_gradients);
        let grad_b_sq = bias_gradients.element_wise_mul(&bias_gradients);

        self.grad_accum_w = &self.grad_accum_w + &grad_w_sq;
        self.grad_accum_b = &self.grad_accum_b + &grad_b_sq;

        let eps = 1e-8;

        let ada_lr_w = weights_gradients
            .element_wise_div(&(&self.grad_accum_w + eps).element_wise_sqrt())
            * self.learning_rate;

        let ada_lr_b = bias_gradients
            .element_wise_div(&(&self.grad_accum_b + eps).element_wise_sqrt())
            * self.learning_rate;

        // ----------------------------
        // 5. Momentum
        // ----------------------------
        self.velocity_w = &self.velocity_w * self.momentum_factor - ada_lr_w;
        self.velocity_b = &self.velocity_b * self.momentum_factor - ada_lr_b;

        (weights + &self.velocity_w, biases + &self.velocity_b)
    }
}
