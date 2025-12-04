use plotters::{
    chart::{ChartBuilder, SeriesLabelPosition},
    prelude::{BitMapBackend, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{BLACK, BLUE, Color as _, IntoFont as _, RED, WHITE},
};

use crate::{
    Dtype,
    callbacks::Callback,
    data_structures::matrix::Matrix,
    networks::network::{Network, TrainingMetric},
};

pub struct PlottingCallback {
    pub metrics: Vec<TrainingMetric>,
    pub output_path: String,
}

impl PlottingCallback {
    pub fn new(output_path: &str) -> Self {
        PlottingCallback {
            metrics: Vec::new(),
            output_path: output_path.to_string(),
        }
    }

    /// Internal method to handle the plotting logic.
    fn plot_metrics_internal(&self) -> anyhow::Result<()> {
        let dir = "./plots";
        std::fs::create_dir_all(dir)?;
        let dir_path = std::path::Path::new(dir);

        let metrics = &self.metrics;
        let output_path = dir_path.join(&self.output_path).to_str().unwrap().to_string();

        if metrics.is_empty() {
            return Err(anyhow::anyhow!("No training data provided for plotting."));
        }

        let max_epoch = metrics.last().map(|m| m.epoch).unwrap_or(0);

        // Loss ranges
        let max_loss = metrics.iter().map(|m| m.loss).fold(Dtype::MIN, Dtype::max);
        let min_loss = metrics.iter().map(|m| m.loss).fold(Dtype::MAX, Dtype::min);

        // === Chart 1: LOSS =======================================================
        {
            let file_path = format!("{}_loss.png", output_path);
            let root = BitMapBackend::new(&file_path, (800, 600)).into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .caption("Training Loss", ("sans-serif", 30).into_font())
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(0..max_epoch, (min_loss * 0.9)..(max_loss * 1.1))?;

            chart
                .configure_mesh()
                .x_desc("Epoch")
                .y_desc("Loss")
                .axis_desc_style(("sans-serif", 15).into_font())
                .draw()?;

            chart
                .draw_series(LineSeries::new(
                    metrics.iter().map(|m| (m.epoch, m.loss)),
                    &RED.mix(0.8),
                ))?
                .label("Loss")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.8)));

            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .border_style(&BLACK)
                .background_style(&WHITE.mix(0.8))
                .draw()?;

            log::info!("Saved loss chart at {}", file_path);
        }

        // === Chart 2: ACCURACY ===================================================
        {
            let file_path = format!("{}_accuracy.png", output_path);
            let root = BitMapBackend::new(&file_path, (800, 600)).into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .caption("Training Accuracy", ("sans-serif", 30).into_font())
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(0..max_epoch, 0.0..1.05 as Dtype)?; // accuracy range

            chart
                .configure_mesh()
                .x_desc("Epoch")
                .y_desc("Accuracy")
                .axis_desc_style(("sans-serif", 15).into_font())
                .draw()?;

            chart
                .draw_series(LineSeries::new(
                    metrics.iter().map(|m| (m.epoch, m.accuracy)),
                    &BLUE.mix(0.8),
                ))?
                .label("Accuracy")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.8)));

            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .border_style(&BLACK)
                .background_style(&WHITE.mix(0.8))
                .draw()?;

            log::info!("Saved accuracy chart at {}", file_path);
        }

        Ok(())
    }
}

impl Callback for PlottingCallback {
    fn on_epoch_end(&mut self, network: &mut Network, y_pred: &Matrix, y_true: &Matrix) -> bool {
        // Calculate metrics using the current predictions stored in the network
        let loss = network.calculate_loss(y_pred, y_true);
        let accuracy = network.calculate_accuracy(y_pred, y_true);

        let metric = TrainingMetric {
            epoch: self.metrics.len() as u32, // Simple epoch counter
            loss,
            accuracy,
        };
        self.metrics.push(metric);
        return false;
    }

    // Now accepts the mutable network reference but ignores it for plotting
    fn on_train_end(&mut self, _network: &mut Network) {
        // Plotting happens automatically when training ends
        if let Err(e) = self.plot_metrics_internal() {
            log::error!("Failed to generate training plot: {}", e);
        }
    }
}
