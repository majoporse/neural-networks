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
        let metrics = &self.metrics;
        let output_path = &self.output_path;

        if metrics.is_empty() {
            return Err(anyhow::anyhow!("No training data provided for plotting."));
        }

        let max_epoch = metrics.last().map(|m| m.epoch).unwrap_or(0);
        let max_loss = metrics.iter().map(|m| m.loss).fold(Dtype::MIN, Dtype::max);
        let min_loss = metrics.iter().map(|m| m.loss).fold(Dtype::MAX, Dtype::min);

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
}

impl Callback for PlottingCallback {
    fn on_epoch_end(&mut self, network: &mut Network, y_pred: &Matrix, y_true: &Matrix) {
        // Calculate metrics using the current predictions stored in the network
        let loss = network.calculate_loss(y_pred, y_true);
        let accuracy = network.calculate_accuracy(y_pred, y_true);

        let metric = TrainingMetric {
            epoch: self.metrics.len() as u32, // Simple epoch counter
            loss,
            accuracy,
        };
        self.metrics.push(metric);
    }

    // Now accepts the mutable network reference but ignores it for plotting
    fn on_train_end(&mut self, _network: &mut Network) {
        // Plotting happens automatically when training ends
        if let Err(e) = self.plot_metrics_internal() {
            log::error!("Failed to generate training plot: {}", e);
        }
    }
}
