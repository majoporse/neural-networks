#[cfg(test)]
mod tests {
    use crate::{Dtype, data_structures::matrix::Matrix};

    #[test]
    fn test_new_and_get_set() {
        let mut m = Matrix::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        
        m.set(0, 0, 1.0);
        m.set(1, 2, 5.0);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 5.0);
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix::new(2, 3);
        m.set(0, 0, 1.0);
        m.set(1, 2, 5.0);

        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(2, 1), 5.0);
    }

    #[test]
    fn test_element_wise_mul() {
        let mut a = Matrix::new(2, 2);
        let mut b = Matrix::new(2, 2);
        a.set(0, 0, 2.0);
        a.set(1, 1, 3.0);
        b.set(0, 0, 4.0);
        b.set(1, 1, 5.0);

        let c = a.element_wise_mul(&b);
        assert_eq!(c.get(0, 0), 8.0);
        assert_eq!(c.get(1, 1), 15.0);
    }

    #[test]
    fn test_matrix_ops() {
        let mut a = Matrix::new(2, 2);
        let mut b = Matrix::new(2, 2);
        a.set(0, 0, 1.0); a.set(0, 1, 2.0);
        a.set(1, 0, 3.0); a.set(1, 1, 4.0);

        b.set(0, 0, 2.0); b.set(0, 1, 0.0);
        b.set(1, 0, 1.0); b.set(1, 1, 2.0);

        let mul = &a * &b;
        assert_eq!(mul.get(0, 0), 4.0);
        assert_eq!(mul.get(0, 1), 4.0);
        assert_eq!(mul.get(1, 0), 10.0);
        assert_eq!(mul.get(1, 1), 8.0);

        let add = &a + &b;
        assert_eq!(add.get(0, 0), 3.0);
        assert_eq!(add.get(1, 1), 6.0);

        let sub = &a - &b;
        assert_eq!(sub.get(0, 0), -1.0);
        assert_eq!(sub.get(1, 1), 2.0);

        let scaled = &a * 2.0;
        assert_eq!(scaled.get(0, 0), 2.0);
        assert_eq!(scaled.get(1, 1), 8.0);
    }

    #[test]
    fn test_shuffle_columns() {
        let mut m = Matrix::new(2, 3);
        // Fill with distinct values so we can track columns
        m.set(0, 0, 1.0); m.set(1, 0, 2.0);
        m.set(0, 1, 3.0); m.set(1, 1, 4.0);
        m.set(0, 2, 5.0); m.set(1, 2, 6.0);

        let indices = vec![2, 0, 1]; // new positions
        m.shuffle_columns(&indices);

        // After shuffling, check that columns moved correctly
        assert_eq!(m.get(0, 0), 5.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(0, 1), 1.0);
        assert_eq!(m.get(1, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 2), 4.0);
    }

    #[test]
    fn test_generate_shuffled_indices() {
        let m = Matrix::new(1, 5);
        let indices = m.generate_shuffled_indices();
        assert_eq!(indices.len(), 5);
        for &i in &indices {
            assert!(i < 5);
        }
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]); // ensures it's a permutation
    }

    #[test]
    fn test_split_into_batches() {
        let mut m = Matrix::new(2, 5);
        for c in 0..5 {
            m.set(0, c, c as Dtype);
            m.set(1, c, c as Dtype + 10.0);
        }

        let batches = m.split_into_batches(2);
        assert_eq!(batches.len(), 3); // last batch smaller

        assert_eq!(batches[0].cols, 2);
        assert_eq!(batches[1].cols, 2);
        assert_eq!(batches[2].cols, 1);

        assert_eq!(batches[0].get(0, 1), 1.0);
        assert_eq!(batches[2].get(1, 0), 14.0);
    }
}
