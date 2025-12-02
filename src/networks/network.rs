use anyhow::Result;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

use indicatif::{ProgressBar, ProgressStyle};

use crate::{Dtype, data_structures::matrix::Matrix, layers::Layer};

// --- Metric Structure for CSV and Plotting ---
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct TrainingMetric {
    pub epoch: u32,
    pub loss: Dtype,
    pub accuracy: Dtype,
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Network {
        Network { layers: Vec::new() }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// Performs the forward pass through all layers.
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    /// Performs the backward pass (gradient descent).
    pub fn backward(&mut self, y_true: &Matrix, learning_rate: Dtype) {
        // The Softmax layer (last layer) takes y_true directly in its backward pass
        // because its derivative is combined with the loss function derivative (Cross-Entropy).

        let last_index = self.layers.len() - 1;
        let mut gradient = self.layers[last_index].backward(y_true, learning_rate);

        // Iterate backward through the rest of the layers
        for i in (0..last_index).rev() {
            gradient = self.layers[i].backward(&gradient, learning_rate);
        }
    }

    /// Calculates the Categorical Cross-Entropy Loss.
    pub fn calculate_loss(&self, y_pred: &Matrix, y_true: &Matrix) -> Dtype {
        let batch_size = y_pred.cols as Dtype;
        let num_classes = y_pred.rows;
        let mut loss = 0.0;

        for c in 0..y_pred.cols {
            for r in 0..num_classes {
                let p = y_pred.get(r, c).max(1e-15); // Clamp for stability
                let t = y_true.get(r, c);
                // Categorical Cross-Entropy L = - sum(t * log(p))
                loss += t * p.ln();
            }
        }

        // Return average negative log-likelihood
        -loss / batch_size
    }

    /// Calculates the classification accuracy.
    pub fn calculate_accuracy(&self, y_pred: &Matrix, y_true: &Matrix) -> Dtype {
        let batch_size = y_pred.cols;
        let num_classes = y_pred.rows;
        let mut correct_predictions = 0;

        for c in 0..batch_size {
            // Find the predicted class (max probability index)
            let mut max_pred_val = -1.0;
            let mut predicted_class = 0;
            for r in 0..num_classes {
                if y_pred.get(r, c) > max_pred_val {
                    max_pred_val = y_pred.get(r, c);
                    predicted_class = r;
                }
            }

            // Find the true class (one-hot encoded index)
            let mut true_class = 0;
            for r in 0..num_classes {
                if y_true.get(r, c) > 0.9 {
                    true_class = r;
                    break;
                }
            }

            if predicted_class == true_class {
                correct_predictions += 1;
            }
        }

        correct_predictions as Dtype / batch_size as Dtype
    }

    pub fn train(
        &mut self,
        input_x: &Matrix,
        y_true: &Matrix,
        learning_rate: Dtype,
        epochs: usize,
    ) -> Result<Vec<TrainingMetric>> {
        // Return metrics vector
        // split the input into batches
        // let BATCH_SIZE = 1;
        // let matrices: Matrix[] = input_x.
        let mut metrics_history = Vec::with_capacity(epochs + 1);

        // --- CSV SETUP (Optional: if you want to save the raw data) ---
        // let file = std::fs::File::create("training_metrics_raw.csv")?;
        // let mut writer = Writer::from_writer(file);
        // writer.write_record(&["epoch", "loss", "accuracy"])?;

        let bar = ProgressBar::new(epochs as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap(),
        );


        for epoch in 1..=epochs {
            // FORWARD PASS
            let y_pred = self.forward(&input_x);
            let loss = self.calculate_loss(&y_pred, &y_true);
            let accuracy = self.calculate_accuracy(&y_pred, &y_true); // Calculate Accuracy

            // BACKWARD PASS & UPDATE
            self.backward(&y_true, learning_rate);

            // --- PROGRESS BAR & METRIC LOGGING ---
            bar.inc(1);

            if epoch % 10 == 0 || epoch == 1 {
                bar.set_message(format!("Loss: {:.6} | Acc: {:.4}", loss, accuracy));
                bar.tick();

                // Record the metric
                let metric = TrainingMetric {
                    epoch: epoch as u32,
                    loss,
                    accuracy,
                };
                metrics_history.push(metric);

                // writer.serialize(&metric)?; // Optional: write to CSV
                // std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        for layer in &self.layers {
            log::info!("weights: {:?}", layer.get_weights());
            log::info!("biases: {:?}", layer.get_biases());
        }


        bar.finish_with_message(format!(
            "Training Complete! Final Loss: {:.6}",
            metrics_history.last().map(|m| m.loss).unwrap_or(0.0)
        ));
        log::info!("Training metrics recorded successfully.");

        Ok(metrics_history)
    }
}

pub fn plot_metrics(metrics: &[TrainingMetric], output_path: &str) -> Result<()> {
    if metrics.is_empty() {
        return Err(anyhow::anyhow!("No training data provided for plotting."));
    }

    // Determine min/max values for scaling the chart axes
    let max_epoch = metrics.last().map(|m| m.epoch).unwrap_or(0);

    let max_loss = metrics
        .iter()
        .map(|m| m.loss)
        .fold(Dtype::MIN, |arg0: Dtype, arg1: Dtype| {
            Dtype::max(arg0, arg1 as Dtype)
        });

    let min_loss = metrics
        .iter()
        .map(|m| m.loss)
        .fold(Dtype::MAX, |arg0: Dtype, arg1: Dtype| {
            Dtype::min(arg0, arg1 as Dtype)
        });

    // 1. Setup the Plotting Backend (PNG file)
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // 2. Configure the Chart
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "NN Training Progress (Loss & Accuracy)",
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        // Primary Y-axis (Loss)
        .build_cartesian_2d(0..max_epoch, (min_loss * 0.9)..(max_loss * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .axis_desc_style(("sans-serif", 15).into_font())
        .draw()?;

    // 3. Draw the Loss Line (Primary Y-axis)
    chart
        .draw_series(LineSeries::new(
            metrics.iter().map(|m| (m.epoch, m.loss)),
            &RED.mix(0.8),
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.8)));

    // 4. Configure the Secondary Y-Axis (Accuracy)
    let mut chart = chart.set_secondary_coord(0.0..1.05, 0.0..1.05); // Accuracy range: 0 to 1.05

    chart.configure_secondary_axes().y_desc("Accuracy").draw()?;

    // 5. Draw the Accuracy Line (Secondary Y-axis)
    chart
        .draw_series(LineSeries::new(
            metrics.iter().map(|m| (m.epoch, m.accuracy)),
            &BLUE.mix(0.8),
        ))?
        .label("Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.8)));

    // 6. Finalize and Save
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    log::info!("Successfully generated plot at: {}", output_path);

    Ok(())
}
