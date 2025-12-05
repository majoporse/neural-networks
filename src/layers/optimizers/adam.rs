use crate::{Dtype, data_structures::matrix::Matrix, layers::optimizers::Optimizer};

pub struct Adam {
    learning_rate: Dtype,
    epsilon: Dtype,

    // First moment (Momentum/Velocity) - Usually denoted 'm'
    m_w: Matrix,
    m_b: Matrix,

    // Second moment (Squared Gradient Accumulation) - Usually denoted 'v'
    v_w: Matrix,
    v_b: Matrix,

    // Decay rates (usually 0.9 and 0.999)
    beta1: Dtype,
    beta2: Dtype,

    // Time step counter for bias correction
    t: Dtype,

    weight_decay: Dtype,
}
// Note: Changed struct name from AdaGrad to Adam

impl Adam {
    pub fn new(
        learning_rate: Dtype,
        beta1: Dtype,   // Typically 0.9
        beta2: Dtype,   // Typically 0.999
        epsilon: Dtype, // Typically 1e-8
        weight_decay: Dtype,
        input_size: usize,
        output_size: usize,
    ) -> Adam {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,

            // Initialize moments (m and v) to zeros
            m_w: Matrix::new(output_size, input_size),
            m_b: Matrix::new(output_size, 1),

            v_w: Matrix::new(output_size, input_size),
            v_b: Matrix::new(output_size, 1),

            t: 0.0, // Initial time step
            weight_decay,
        }
    }
}

// https://github.com/theroyakash/Adam/blob/master/src/Screen%20Shot%202020-02-05%20at%2010.23.14%20PM.png

impl Optimizer for Adam {
    fn update(
        &mut self,
        weights: &Matrix,
        biases: &Matrix,
        mut weights_gradients: Matrix,
        bias_gradients: Matrix,
    ) -> (Matrix, Matrix) {
        // 1. Time Step Increment
        self.t += 1.0;
        let t_us = self.t;

        // 2. Weight decay (L2) - Applied to the gradient
        if self.weight_decay > 0.0 {
            let l2_grad_w = &*weights * self.weight_decay;
            weights_gradients = &weights_gradients + &l2_grad_w;
        }

        // --- 3. Moment Estimates (m and v) ---
        // New gradients (g) are weights_gradients and bias_gradients
        let g_w_sq = weights_gradients.element_wise_mul(&weights_gradients);
        let g_b_sq = bias_gradients.element_wise_mul(&bias_gradients);

        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        self.m_w = (&self.m_w * self.beta1) + (&weights_gradients * (1.0 - self.beta1));
        self.m_b = (&self.m_b * self.beta1) + (&bias_gradients * (1.0 - self.beta1));

        // v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t^2)
        self.v_w = (&self.v_w * self.beta2) + (&g_w_sq * (1.0 - self.beta2));
        self.v_b = (&self.v_b * self.beta2) + (&g_b_sq * (1.0 - self.beta2));

        // --- 4. Bias Correction ---
        // m_hat = m_t / (1 - beta1^t)
        let beta1_t = self.beta1.powf(t_us);
        let beta2_t = self.beta2.powf(t_us);

        let m_hat_w = &self.m_w / (1.0 - beta1_t);
        let m_hat_b = &self.m_b / (1.0 - beta1_t);

        // v_hat = v_t / (1 - beta2^t)
        let v_hat_w = &self.v_w / (1.0 - beta2_t);
        let v_hat_b = &self.v_b / (1.0 - beta2_t);

        // --- 5. Parameter Update ---
        // theta_t+1 = theta_t - LR * [ m_hat / (sqrt(v_hat) + epsilon) ]
        let lr = self.learning_rate;
        let eps = self.epsilon;

        // Weights
        let denom_w = &v_hat_w.element_wise_sqrt() + eps;
        let step_w = (m_hat_w.element_wise_div(&denom_w)) * lr;

        // Biases
        let denom_b = &v_hat_b.element_wise_sqrt() + eps;
        let step_b = (m_hat_b.element_wise_div(&denom_b)) * lr;

        (weights - &step_w, biases - &step_b)
    }
}
