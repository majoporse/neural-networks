use rand::{Rng, rng};
 // Import Rng, thread_rng, and SeedableRng
use std::ops::{Add, Mul, Sub};

use crate::Dtype;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Dtype>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a new matrix with random values initialized using the Kaiming/He
    /// initialization scale for ReLU (or a similar common initialization).
    /// Uses the default thread-local RNG (non-seedable).
    pub fn new_random(rows: usize, cols: usize) -> Matrix {
        let seed = rng().random::<u64>();
        Matrix::new_seeded_random(rows, cols, seed)
    }

    /// Creates a new matrix with random values using a fixed seed.
    /// This is essential for reproducible training runs.
    pub fn new_seeded_random(rows: usize, cols: usize, seed: u64) -> Matrix {
        // Pcg64 is a good, fast, and deterministic RNG for seeded use.

        let mut rand = rand_simple::Normal::new([seed as u32, seed as u32]);
        let std = 2.0 / rows as Dtype + 1.0;
        rand.try_set_params(0.0, std as f64).unwrap();

        let data = (0..rows * cols)
            .map(|_| rand.sample() as Dtype)
            .collect();

        Matrix { rows, cols, data }
    }

    pub fn get(&self, r: usize, c: usize) -> Dtype {
        // Access data in column-major order (r + c * rows)
        self.data[r + c * self.rows]
    }

    pub fn set(&mut self, r: usize, c: usize, val: Dtype) {
        // Set data in column-major order (r + c * rows)
        self.data[r + c * self.rows] = val;
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result.set(col, row, self.get(row, col));
            }
        }
        result
    }

    pub fn element_wise_mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    pub fn split_into_batches(){
        
    }
}

// --- Operator Overloads ---

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrix multiplication dimensions must match."
        );

        let mut result = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }

                result.set(i, j, sum);
            }
        }
        result
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }
}

impl Mul<Dtype> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: Dtype) -> Matrix {
        let mut result = self.clone();
        for val in result.data.iter_mut() {
            *val *= scalar;
        }
        result
    }
}

pub fn sum_cols(matrix: &Matrix) -> Matrix {
    let mut result = Matrix::new(matrix.rows, 1);
    for row in 0..matrix.rows {
        let mut sum = 0.0;
        for col in 0..matrix.cols {
            sum += matrix.get(row, col);
        }
        result.set(row, 0, sum);
    }
    result
}
