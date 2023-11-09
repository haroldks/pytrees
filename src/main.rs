#![allow(unused)]
#![warn(clippy::too_many_arguments)]

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
use crate::algorithms::dl85_utils::structs_enums::HasIntersected::No;
use clap::Parser;
use rayon::prelude::*;
use rayon::prelude::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
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

    /// Time limit
    #[arg(long, default_value_t = 600)]
    time_limit: usize,

    /// Max error
    #[arg(long, default_value_t = <usize>::MAX)]
    max_error: usize,
}

fn main() {
    let args = Args::parse();
    if !args.file.exists() {
        panic!("File does not exist");
    }

    let file = args.file.to_str().unwrap();
    let depth = args.depth;
    let min_sup = args.support;
    let time_limit = args.time_limit;
    let max_error = args.max_error;

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
        0 => Box::<NoHeuristic>::default(),
        1 => Box::<GiniIndex>::default(),
        2 => Box::<InformationGain>::default(),
        3 => Box::<InformationGainRatio>::default(),
        _ => {
            println!("Invalid heuristic type");
            process::exit(1);
        }
    };

    let dataset = BinaryDataset::load(file, false, 0.0);
    let bitset = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset);

    let mut algo: DL85<'_, _, Data> = DL85::new(
        min_sup,
        depth,
        max_error,
        time_limit,
        specialization,
        lower_bound,
        branching,
        CacheInit::Normal,
        0,
        false,
        heuristic.as_mut(),
        None,
        None,
    );
    algo.fit(&mut structure);
    println!("--------------- Search Tree ---------------");
    algo.tree.print();
    println!("--------------- Search Tree ---------------");
}
