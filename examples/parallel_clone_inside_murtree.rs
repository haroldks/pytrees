use clap::Parser;
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
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, to_writer};

#[derive(Serialize, Deserialize)]
struct ResStructCloned {
    depth: Vec<usize>,
    runtimes: Vec<Vec<f64>>,
    errors: Vec<Vec<usize>>,
}

/// Simple program to greet a person
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Test File path
    #[arg(short, long)]
    file: PathBuf,
}

impl ResStructCloned {
    pub fn to_json(&self, filename: String) -> Result<(), Error> {
        if let Err(e) = to_writer(&File::create(filename)?, &self) {
            println!("File Creating error: {}", e.to_string());
        };
        Ok(())
    }
}

fn main() {
    let args = Args::parse();
    if !args.file.exists() {
        panic!("File does not exist");
    }
    let file = args.file.to_str().unwrap();
    // Get file name without extension
    let filename = args.file.file_stem().unwrap().to_str().unwrap().to_string();
    let res_file = format!("{}_run_with_mutree_split_in_threads.json", filename);

    let dataset = BinaryDataset::load(file, false, 0.0);
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
        println!("Depth: {}", depth);
        for num_thread in 1..num_threads + 1 {
            println!("\tNum threads: {}", num_thread);
            let mut duration = 0.0;
            for rep in 0..n_repeat {
                let mut structure = RSparseBitsetStructure::new(&bitset, num_thread);
                let start = Instant::now();
                let tree = LGDT::fit(&mut structure, 1, depth, MurTree::fit);
                duration += start.elapsed().as_secs_f64() * 1000.0;
                if rep == n_repeat - 1 {
                    depth_errors.push(LGDT::get_tree_error(&tree));
                }
            }
            depth_runtimes.push(duration / n_repeat as f64);
        }
        runtimes.push(depth_runtimes);
        errors.push(depth_errors);
    }

    let res = ResStructCloned {
        depth: depths,
        runtimes: runtimes,
        errors: errors,
    };
    res.to_json(res_file).expect("Error writing file");
}
