 

use anyhow::anyhow; // Import anyhow! macro for easy error creation

use crate::data_structures::matrix::Matrix;

pub fn load_data(
    x_path: &str,
    y_path: &str,
    input_size: usize,
    output_size: usize,
) -> anyhow::Result<(Matrix, Matrix)> {
    log::info!("Reading input from: {} and labels from: {}", x_path, y_path);

    // X Data Loading
    let file_x = std::fs::File::open(x_path)?;
    let mut rdr_x = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_x);

    let records_x: Vec<csv::StringRecord> = rdr_x.records().collect::<Result<_, _>>()?;
    log::info!("Loaded {} records from X CSV.", records_x.len());
    
    let sample_count = records_x.len();

    if sample_count == 0 {
        return Err(anyhow!("No data found in X CSV at path: {}", x_path));
    }

    let mut input_x = Matrix::new(input_size, sample_count);

    for (i, record) in records_x.into_iter().enumerate() {
        log::info!("Processing X Row {}: {:?}", i, record);
        if record.len() != input_size {
            return Err(anyhow!(
                "X Row {} has wrong column count: expected {}, got {}",
                i,
                input_size,
                record.len(),
            ));
        }
        for c in 0..input_size {
            input_x.set(c, i, (&record[c]).parse().map_err(anyhow::Error::from)?);
        }
    }

    // Y Data Loading
    let file_y = std::fs::File::open(y_path)?;
    let mut rdr_y = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file_y); 

    let records_y: Vec<csv::StringRecord> = rdr_y.records().collect::<Result<_, _>>()?;
    if records_y.len() != sample_count {
        log::error!("Mismatched number of samples between X and Y files: X has {}, Y has {}", sample_count, records_y.len());
        return Err(anyhow!(
            "Mismatched number of samples between X and Y files: X has {}, Y has {}", 
            sample_count, 
            records_y.len()
        ));
    }

    let mut y_true = Matrix::new(output_size, sample_count);

    for (i, record) in records_y.into_iter().enumerate() {
        if record.len() != 1 {
            return Err(anyhow!(
                "Y Label Row {} expected 1 column, got {}", 
                i, 
                record.len()
            ));
        }

        let class_index: usize = (&record[0]).parse().map_err(anyhow::Error::from)?;

        if class_index >= output_size {
            return Err(anyhow!(
                "Invalid label index {}. Expected index < {} (output_size).",
                class_index, 
                output_size
            ));
        }

        y_true.set(class_index, i, 1.0);
    }

    Ok((input_x, y_true))
}