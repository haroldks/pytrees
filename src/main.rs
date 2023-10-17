#![allow(unused)]

use crate::algorithms::algorithm_trait::Algorithm;
use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::{
    BranchingType, CacheInit, DiscrepancyStrategy, LowerBoundHeuristic, Specialization,
};
use crate::algorithms::idk::IDK;
use crate::algorithms::info_gain::InfoGain;
use crate::algorithms::lds_dl85::LDSDL85;
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::structures::caching::trie::{Data, TrieNode};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use itertools::Itertools;
use ndarray::s;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{process, thread};
// use rayon::iter::IntoParallelIterator;
use clap::Parser;
use rayon::prelude::*;
use rayon::prelude::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json;
use serde_json::to_writer;
use std::fs::File;
use std::io::Error;
use std::path::PathBuf;

mod algorithms;
mod dataset;
mod heuristics;
mod post_process;
mod structures;

#[derive(Debug, Serialize, Deserialize)]
struct ExpeRes {
    size: Vec<usize>,
    res: Vec<Vec<f64>>,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Test File path
    #[arg(short, long)]
    file: PathBuf,

    /// Maximum depth
    #[arg(short, long)]
    depth: usize,

    /// Minimum support
    #[arg(short, long, default_value_t = 1)]
    support: usize,

    /// Use Murtree Spacialization Algorithm
    #[arg(short, long)]
    use_specialization: bool,

    /// Lower bound heuristic
    /// 0: None
    /// 1: Similarity
    #[arg(short, long, default_value_t = 0)]
    lower_bound_heuristic: usize,

    /// Branching type
    /// 0: None
    /// 1: Dynamic
    #[arg(short, long, default_value_t = 0)]
    branching_type: usize,

    /// Sorting heuristic
    /// 0: None
    /// 1: Gini
    /// 2: Information Gain
    /// 3: Information Gain Ratio
    #[arg(long, default_value_t = 0)]
    sorting_heuristic: usize,
}

impl ExpeRes {
    pub fn to_json(&self, filename: String) -> Result<(), Error> {
        if let Err(e) = to_writer(&File::create(filename)?, &self) {
            println!("File Creating error: {}", e.to_string());
        };
        Ok(())
    }
}

fn gen_random_vec(size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut a = vec![];
    for _ in 0..size {
        a.push(rng.gen_range(0..1_000_000) as usize)
    }
    a
}

fn compare(a: &[usize], b: &[usize], n_threads: usize) -> usize {
    let mut rng = rand::thread_rng();

    let size = a.len();

    if n_threads <= 1 {
        let mut value = 0;
        for i in 0..size {
            value += (a[i] & b[i]).count_ones();
        }
        return value as usize;
    }

    let chunk_size = size / n_threads;
    let mut value = 0;
    rayon::scope(|s| {
        let val = Arc::new(Mutex::new(&mut value));
        for i in 0..n_threads {
            // let chunk = &indexes[chunk_start..chunk_end];
            let thread_val = val.clone();
            let res = s.spawn(move |_| {
                let chunk_start = i * chunk_size;
                let chunk_end = if i == n_threads - 1 {
                    size
                } else {
                    (i + 1) * chunk_size
                };

                let mut val = thread_val.lock().unwrap();
                let mut count = 0;
                for idx in chunk_start..chunk_end {
                    count += (a[idx] & b[idx]).count_ones();
                }
                **val += count;
            });
        }
    });
    return value as usize;
}

fn compare_free_contention(a: &[usize], b: &[usize], n_threads: usize) -> usize {
    let size = a.len();

    if n_threads <= 1 {
        let mut value = 0;
        for i in 0..size {
            value += (a[i] & b[i]).count_ones();
        }
        return value as usize;
    }

    let chunk_size = size / n_threads;
    let mut values = vec![0; n_threads];

    values.par_iter_mut().enumerate().for_each(|(i, val)| {
        let chunk_start = i * chunk_size;
        let chunk_end = if i == n_threads - 1 {
            size
        } else {
            (i + 1) * chunk_size
        };

        let mut count = 0;
        for idx in chunk_start..chunk_end {
            count += (a[idx] & b[idx]).count_ones();
        }
        *val = count;
    });

    let mut value = 0;
    for val in values.iter() {
        value += *val;
    }

    value as usize
}

fn main() {
    let args = Args::parse();
    if !args.file.exists() {
        panic!("File does not exist");
    }

    let file = args.file.to_str().unwrap();
    let depth = args.depth;
    let min_sup = args.support;

    let use_specialization = args.use_specialization;
    let lower_bound_heuristic = args.lower_bound_heuristic;
    let branching_type = args.branching_type;
    let sorting_heuristic = args.sorting_heuristic;

    let specialization = match use_specialization {
        true => Specialization::Murtree,
        false => Specialization::None,
    };

    let lower_bound = match lower_bound_heuristic {
        0 => LowerBoundHeuristic::None,
        1 => LowerBoundHeuristic::Similarity,
        _ => {
            println!("Invalid lower bound heuristic");
            process::exit(1);
        }
    };

    let branching = match branching_type {
        0 => BranchingType::None,
        1 => BranchingType::Dynamic,
        _ => {
            println!("Invalid branching type");
            process::exit(1);
        }
    };

    let mut heuristic: Box<dyn Heuristic> = match sorting_heuristic {
        0 => Box::new(NoHeuristic::default()),
        1 => Box::new(GiniIndex::default()),
        2 => Box::new(InformationGain::default()),
        3 => Box::new(InformationGainRatio::default()),
        _ => {
            println!("Invalid heuristic type");
            process::exit(1);
        }
    };

    let dataset = BinaryDataset::load(file, false, 0.0);
    let bitset = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset, 1);

    let mut algo: DL85<'_, _, Data> = DL85::new(
        min_sup,
        depth,
        <usize>::MAX,
        <usize>::MAX,
        specialization,
        lower_bound,
        branching,
        CacheInit::Normal,
        0,
        false,
        heuristic.as_mut(),
    );
    algo.fit(&mut structure);
    algo.tree.print();
}
