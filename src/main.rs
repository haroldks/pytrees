#![allow(unused)]

use crate::algorithms::algorithm_trait::Algorithm;
use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::Specialization::Murtree;
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
use std::time::Instant;

mod algorithms;
mod dataset;
mod heuristics;
mod post_process;
mod structures;

fn main() {
    // let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
    let dataset = BinaryDataset::load(
        "experiments/data/parallel_datasets/275_1500000.csv",
        false,
        0.0,
    );
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);
    let num_attributes = structure.num_attributes();
    println!("Num attributes: {:?}", num_attributes);
    println!("Num labels: {:?}", structure.num_labels());
    println!("Support: {:?}", structure.support());
    let n = 10;
    let start = Instant::now();
    let n = 100;
    for _ in 0..n {
        structure.parallel_temp_push((20, 1));
    }
    let duration = start.elapsed().as_micros() as f64 / n as f64;
    println!("Parallel temp push time: {:?} us", duration);

    let start = Instant::now();
    for _ in 0..n {
        structure.temp_push((20, 1));
    }
    let duration = start.elapsed().as_micros() as f64 / n as f64;
    println!("Sequential temp push time: {:?} us", duration);

    // let mut heuristic: Box<dyn Heuristic> = Box::new(NoHeuristic::default());

    // let mut algo: DL85<'_, _, Data> = DL85::new(
    //     1,
    //     2,
    //     <usize>::MAX,
    //     600,
    //     Specialization::None,
    //     LowerBoundHeuristic::None,
    //     BranchingType::None,
    //     CacheInit::WithMemoryDynamic,
    //     0,
    //     true,
    //     heuristic.as_mut(),
    // );

    // let algo = LGDT::fit(&mut structure, 5, 2, InfoGain::fit);
    // algo.print();
    // algo.fit(&mut structure);
    // algo.tree.print();
}
