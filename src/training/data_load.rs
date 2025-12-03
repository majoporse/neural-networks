use anyhow::anyhow;

use crate::{Dtype, data_structures::matrix::Matrix}; // Assuming Matrix is defined elsewhere

// NOTE: ROW = FEATURE INDEX \ COLUMN = SAMPLE INDEX
pub fn load_data(
    x_path: &str,
    y_path: &str,
    input_size: usize,
    output_size: usize,
) -> anyhow::Result<(Matrix, Matrix)> {
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

    let file_y = std::fs::File::open(y_path)?;
    let mut rdr_y = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_y);

    let records_y: Vec<csv::StringRecord> = rdr_y.records().collect::<Result<_, _>>()?;

    if records_y.len() != sample_count {
        log::error!(
            "Mismatched number of samples between X and Y files: X has {}, Y has {}",
            sample_count,
            records_y.len()
        );
        return Err(anyhow!(
            "Mismatched number of samples between X and Y files: X has {}, Y has {}",
            sample_count,
            records_y.len()
        ));
    }

    let mut inputs = Matrix::new(input_size, records_x.len());
    let mut labels = Matrix::new(output_size, records_y.len());

    for (i, (x_chunk, y_chunk)) in records_x.iter().zip(records_y.iter()).enumerate() {
        if x_chunk.len() != input_size {
            return Err(anyhow!(
                "X record in batch {} has wrong column count: expected {}, got {}",
                i,
                input_size,
                x_chunk.len(),
            ));
        }

        if y_chunk.len() != 1 {
            return Err(anyhow!(
                "Y Label record in batch {} expected 1 column, got {}",
                i,
                y_chunk.len()
            ));
        }

        for feature_index in 0..input_size {
            let value: Dtype = (&x_chunk[feature_index])
                .parse()
                .map_err(anyhow::Error::from)?;
            inputs.set(feature_index, i, value);
        }

        // --- Populate Y Batch Matrix (One-Hot Encoded) ---
        let class_index: usize = (&y_chunk[0]).parse().map_err(anyhow::Error::from)?;

        if class_index >= output_size {
            return Err(anyhow!(
                "Invalid label index {}. Expected index < {} (output_size).",
                class_index,
                output_size
            ));
        }
        labels.set(class_index, i, 1.0);
        }

    log::info!(
        "Successfully loaded {} records from X and {} records from Y.",
        records_x.len(),
        records_y.len()
    );

    Ok((inputs, labels))
}
