#![allow(unused)]
use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::{
    BranchingType, LowerBoundHeuristic, Specialization,
};
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::structures::caching::trie::Data;
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;

mod algorithms;
mod dataset;
mod heuristics;
mod post_process;
mod structures;

fn main() {
    let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);

    let mut heuristic: Box<dyn Heuristic> = Box::new(NoHeuristic::default());

    let mut algo: DL85<'_, _, Data> = DL85::new(
        1,
        4,
        <usize>::MAX,
        <usize>::MAX,
        Specialization::None,
        LowerBoundHeuristic::None,
        BranchingType::None,
        true,
        heuristic.as_mut(),
    );
    algo.fit(&mut structure);

    println!("Statistics: {:?}", algo.statistics);
    algo.tree.print();
}
