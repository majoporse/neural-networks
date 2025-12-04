use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{Dtype, grid_search::minst_config::train_mnist_with_config};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub hidden_size: usize,
    pub hidden_size_2: usize,
    pub learning_rate: Dtype,
    pub batch_size: usize,
    pub momentum: Dtype,
    pub weight_decay: Dtype,
    pub epochs: usize,
}

pub fn run_grid_search() -> anyhow::Result<()> {
    let learning_rates = vec![0.003, 0.007, 0.005];
    let batch_sizes = vec![64];
    let hidden_sizes = vec![128, 64];
    let hidden_sizes_2 = vec![64, 32];
    let weight_decays = vec![0.0, 0.0001, 0.01];
    let momenta = vec![0.01, 0.001];
    let epochs = 10;

    // create config list
    let mut configs = Vec::new();
    for &lr in &learning_rates {
        for &bs in &batch_sizes {
            for &hs in &hidden_sizes {
                for &wd in &weight_decays {
                    for &mom in &momenta {
                        for &hs2 in &hidden_sizes_2 {
                            configs.push(TrainConfig {
                                learning_rate: lr,
                                batch_size: bs,
                                hidden_size: hs,
                                hidden_size_2: hs2,
                                weight_decay: wd,
                                momentum: mom,
                                epochs,
                            });
                        }
                    }
                }
            }
        }
    }

    log::info!("Running {} configurations in parallel…", configs.len());

    // Parallel execution using rayon:
    let results: Vec<_> = configs
        .par_iter()
        .map(|cfg| {
            let res = train_mnist_with_config(cfg);
            match res {
                Ok(acc) => (cfg.clone(), acc),
                Err(e) => {
                    log::error!("Config {:?} failed: {:?}", cfg, e);
                    (cfg.clone(), -1.0)
                }
            }
        })
        .collect();

    // sort best first
    let mut results = results;
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    log::info!("===== GRID SEARCH RESULTS =====");
    for (cfg, acc) in &results {
        log::info!("Acc {:5.2}% → {:?}", acc * 100.0, cfg);
    }

    let (best_cfg, best_acc) = &results[0];
    log::info!("===== BEST CONFIG =====");
    log::info!("Accuracy: {:5.2}%", best_acc * 100.0);
    log::info!("Config: {:?}", best_cfg);

    // save best config to file
    let best_cfg_json = serde_json::to_string_pretty(&best_cfg)?;
    std::fs::write("best_train_config.json", best_cfg_json)?;
    // save all configs and results to file in descending order
    let all_results_json = serde_json::to_string_pretty(&results)?;
    std::fs::write("all_train_results.json", all_results_json)?;
    Ok(())
}
