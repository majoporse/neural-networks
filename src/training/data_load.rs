use anyhow::anyhow;

use crate::{Dtype, data_structures::matrix::Matrix}; // Assuming Matrix is defined elsewhere

// Note: The return type is now Vec<Matrix> for both X and Y, 
// where each Matrix is a batch of (input_size, batch_size)
pub fn load_data(
    x_path: &str,
    y_path: &str,
    input_size: usize,
    output_size: usize,
    batch_size: usize,
) -> anyhow::Result<(Vec<Matrix>, Vec<Matrix>)> {
    log::info!("Reading input from: {} and labels from: {}", x_path, y_path);

    // --- X Data Loading: Load all records ---
    let file_x = std::fs::File::open(x_path)?;
    let mut rdr_x = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_x);

    // Collect all records into a single Vec
    let records_x: Vec<csv::StringRecord> = rdr_x.records().collect::<Result<_, _>>()?;
    log::info!("Loaded {} records from X CSV.", records_x.len());
    
    let sample_count = records_x.len();

    if sample_count == 0 {
        return Err(anyhow!("No data found in X CSV at path: {}", x_path));
    }
    
    if batch_size == 0 {
        return Err(anyhow!("batch_size must be greater than 0."));
    }

    // --- Y Data Loading: Load all records ---
    let file_y = std::fs::File::open(y_path)?;
    let mut rdr_y = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_y); 

    let records_y: Vec<csv::StringRecord> = rdr_y.records().collect::<Result<_, _>>()?;
    
    // Check for mismatched counts
    if records_y.len() != sample_count {
        log::error!("Mismatched number of samples between X and Y files: X has {}, Y has {}", sample_count, records_y.len());
        return Err(anyhow!(
            "Mismatched number of samples between X and Y files: X has {}, Y has {}", 
            sample_count, 
            records_y.len()
        ));
    }

    // --- BATCHING LOGIC ---

    let mut batched_x = Vec::new();
    let mut batched_y = Vec::new();
    
    for (batch_index, (x_chunk, y_chunk)) in records_x
        .chunks(batch_size) // Groups the records into slices of batch_size
        .zip(records_y.chunks(batch_size)) // Zip with Y records
        .enumerate()
    {
        let current_batch_size = x_chunk.len();
        
        let mut x_batch_matrix = Matrix::new(input_size, current_batch_size);
        let mut y_batch_matrix = Matrix::new(output_size, current_batch_size);

        for (sample_index_in_batch, (x_record, y_record)) in x_chunk.iter().zip(y_chunk.iter()).enumerate() {
            
            if x_record.len() != input_size {
                return Err(anyhow!(
                    "X record in batch {} has wrong column count: expected {}, got {}",
                    batch_index,
                    input_size,
                    x_record.len(),
                ));
            }
            
            if y_record.len() != 1 {
                 return Err(anyhow!(
                    "Y Label record in batch {} expected 1 column, got {}", 
                    batch_index, 
                    y_record.len()
                ));
            }
            
            for feature_index in 0..input_size {
                let value: Dtype = (&x_record[feature_index]).parse().map_err(anyhow::Error::from)?;
                // Set value at (row=feature_index, col=sample_index_in_batch)
                x_batch_matrix.set(feature_index, sample_index_in_batch, value); 
            }
            
            // --- Populate Y Batch Matrix (One-Hot Encoded) ---
            let class_index: usize = (&y_record[0]).parse().map_err(anyhow::Error::from)?;

            if class_index >= output_size {
                 return Err(anyhow!(
                    "Invalid label index {}. Expected index < {} (output_size).",
                    class_index, 
                    output_size
                ));
            }
            // Set 1.0 at (row=class_index, col=sample_index_in_batch)
            y_batch_matrix.set(class_index, sample_index_in_batch, 1.0);
        }

        batched_x.push(x_batch_matrix);
        batched_y.push(y_batch_matrix);
    }
    
    log::info!("Successfully created {} batches of size up to {}.", batched_x.len(), batch_size);
    Ok((batched_x, batched_y))
}