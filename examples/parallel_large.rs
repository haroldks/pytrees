use pytrees::algorithms::algorithm_trait::{Algorithm, Basic};
use pytrees::algorithms::lgdt::LGDT;
use pytrees::algorithms::murtree::MurTree;
use pytrees::dataset::binary_dataset::BinaryDataset;
use pytrees::dataset::data_trait::Dataset;
use pytrees::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use pytrees::structures::structure_trait::Structure;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Error;
use std::time::Instant;

use serde_json::{json, to_writer};

#[derive(Serialize, Deserialize)]
struct Results {
    depth: Vec<usize>,
    runtimes: Vec<Vec<f64>>,
    errors: Vec<Vec<usize>>,
}

impl Results {
    pub fn to_json(&self, filename: String) -> Result<(), Error> {
        if let Err(e) = to_writer(&File::create(filename)?, &self) {
            println!("File Creating error: {}", e.to_string());
        };
        Ok(())
    }
}

fn main() {
    let dataset = BinaryDataset::load(
        "experiments/data/parallel_datasets/10_150000000_v.csv",
        false,
        0.0,
    );
    let bitset = RSparseBitsetStructure::format_input_data(&dataset);
    let mut runtimes = vec![];
    let mut errors = vec![];
    let num_threads = 10;
    let num_depths = 8;
    let n_repeat = 5;
    let mut depths = vec![];
    for depth in 2..num_depths + 1 {
        let mut depth_runtimes = vec![];
        let mut depth_errors = vec![];
        depths.push(depth);
        println!("Current depth : {}", depth);
        for num_thread in 1..num_threads + 1 {
            println!("\tNum threads: {}", num_thread);
            let mut duration = 0.0;
            for rep in 0..n_repeat {
                let mut structure = RSparseBitsetStructure::new(&bitset, num_thread);
                let start = Instant::now();
                let tree = LGDT::fit(&mut structure, 1, depth, MurTree::fit);
                duration += start.elapsed().as_secs_f64() * 1000.0;
                let error = LGDT::get_tree_error(&tree);
                if rep == n_repeat - 1 {
                    depth_errors.push(error)
                }
            }
            depth_runtimes.push(duration / n_repeat as f64);

            // depth_errors.push(error);
        }
        runtimes.push(depth_runtimes);
        errors.push(depth_errors);
    }

    let results = Results {
        depth: depths,
        runtimes,
        errors,
    };
    results
        .to_json("results_hybrid_15M.json".to_string())
        .expect("Error writing to file");
}