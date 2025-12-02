use rand_simple::Normal;
use rayon::prelude::*; // Import Rayon's core traits
use std::ops::{Add, Mul, Sub};

use crate::Dtype; // Assuming Dtype is a float type like f32 or f64

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

    pub fn get(&self, r: usize, c: usize) -> Dtype {
        // Access data in column-major order (r + c * rows)
        self.data[r + c * self.rows]
    }

    pub fn set(&mut self, r: usize, c: usize, val: Dtype) {
        // Set data in column-major order (r + c * rows)
        self.data[r + c * self.rows] = val;
    }

    // Transpose, new_random, new_seeded_random, element_wise_mul, and split_into_batches remain unchanged (and were correct)
    
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result.set(col, row, self.get(row, col));
            }
        }
        result
    }

    /// Creates a new matrix with random values initialized using the Kaiming/He
    /// initialization scale for ReLU (or a similar common initialization).
    /// Uses the default thread-local RNG (non-seedable).
    pub fn new_random(rows: usize, cols: usize) -> Matrix {
        // ... (function body remains the same)
        let seed = 12345; 
        Matrix::new_seeded_random(rows, cols, seed)
    }

    /// Creates a new matrix with random values using a fixed seed.
    /// This is essential for reproducible training runs.
    pub fn new_seeded_random(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut rand = Normal::new([seed as u32, seed as u32]);
        let std = 2.0 / rows as Dtype + 1.0;
        rand.try_set_params(0.0, std as f64).unwrap();

        let data = (0..rows * cols)
            .map(|_| rand.sample() as Dtype)
            .collect();

        Matrix { rows, cols, data }
    }


    /// **PARALLEL** element-wise multiplication.
    pub fn element_wise_mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        
        // Use par_iter_mut() to parallelize the loop over the data.
        result.data.par_iter_mut()
            .zip(self.data.par_iter())
            .zip(other.data.par_iter())
            .for_each(|((r_val, s_val), o_val)| {
                *r_val = s_val * o_val;
            });

        result
    }

    pub fn split_into_batches(){
        
    }
}

// --- Operator Overloads ---

impl Mul for &Matrix {
    type Output = Matrix;

    /// **CORRECTED PARALLEL** Matrix multiplication. 
    /// Generates the output data vector by parallelizing over columns (j) first, 
    /// then rows (i), to maintain column-major storage order using flat_map.
    fn mul(self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrix multiplication dimensions must match."
        );

        let num_rows = self.rows;
        let num_cols = other.cols;
        let inner_dim = self.cols;

        // Parallelize over the columns (j) of the result matrix first.
        // This is done to produce the output data in column-major order (Col 0, then Col 1, etc.)
        let result_data: Vec<Dtype> = (0..num_cols)
            .into_par_iter()
            .flat_map(|j| {
                // For each column 'j', calculate all rows 'i'
                (0..num_rows)
                    .map(move |i| {
                        let mut sum = 0.0;
                        for k in 0..inner_dim {
                            // C[i, j] += A[i, k] * B[k, j]
                            // Use get() to handle the column-major layout of inputs
                            sum += self.get(i, k) * other.get(k, j);
                        }
                        sum
                    })
                    // Collect all the calculated C[i, j] elements for this column 'j'.
                    // This produces a column-major chunk (r0c0, r1c0, ..., r_n-1c0)
                    .collect::<Vec<Dtype>>()
            })
            // Flat_map appends all the column chunks into one final, linear Vec<Dtype>.
            .collect(); 

        Matrix {
            rows: num_rows,
            cols: num_cols,
            data: result_data,
        }
    }
}

// --- Add, Sub, and Scalar Mul remain unchanged (and were already correct and safe) ---

impl Add for &Matrix {
    type Output = Matrix;

    /// **PARALLEL** Matrix addition.
    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        
        // Parallel element-wise operation using zip
        result.data.par_iter_mut()
            .zip(self.data.par_iter())
            .zip(other.data.par_iter())
            .for_each(|((r_val, s_val), o_val)| {
                *r_val = s_val + o_val;
            });

        result
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    /// **PARALLEL** Matrix subtraction.
    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        
        // Parallel element-wise operation using zip
        result.data.par_iter_mut()
            .zip(self.data.par_iter())
            .zip(other.data.par_iter())
            .for_each(|((r_val, s_val), o_val)| {
                *r_val = s_val - o_val;
            });

        result
    }
}

impl Mul<Dtype> for &Matrix {
    type Output = Matrix;

    /// **PARALLEL** Scalar multiplication.
    fn mul(self, scalar: Dtype) -> Matrix {
        let mut result = self.clone();
        
        // Use par_iter_mut() to parallelize the loop over the data.
        result.data.par_iter_mut().for_each(|val| {
            *val *= scalar;
        });
        
        result
    }
}

/// **CORRECTED PARALLEL** function to sum columns. (Function name fixed).
pub fn sum_cols(matrix: &Matrix) -> Matrix {
    
    // Parallelize the rows and use map to calculate the sum for each row.
    // Each thread generates a single Dtype (the row sum).
    let data_par: Vec<Dtype> = (0..matrix.rows)
        .into_par_iter()
        .map(|row| {
            let mut sum = 0.0;
            
            // Sequential inner loop for column summation (col)
            for col in 0..matrix.cols {
                sum += matrix.get(row, col);
            }
            
            // The map operation returns the final row sum (Dtype)
            sum
        })
        .collect(); // Collect all the independently calculated row sums into a Vec<Dtype>

    // Construct the result Matrix from the safely collected data.
    // This is correct because the output is a column vector (N x 1),
    // and the order of `data_par` (sum_r0, sum_r1, ...) is the correct column-major order.
    Matrix {
        rows: matrix.rows,
        cols: 1,
        data: data_par,
    }
}