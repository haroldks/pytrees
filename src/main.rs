#![allow(unused)]
use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::{
    BranchingType, CacheInit, DiscrepancyStrategy, LowerBoundHeuristic, Specialization,
};
use crate::algorithms::lds_dl85::LDSDL85;
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::structures::caching::trie::{Data, TrieNode};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;

mod algorithms;
mod dataset;
mod heuristics;
mod post_process;
mod structures;

fn main() {
    let dataset = BinaryDataset::load("test_data/splice-1.txt", false, 0.0);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);

    let mut heuristic: Box<dyn Heuristic> = Box::new(InformationGain::default());

    let mut algo: LDSDL85<'_, _, Data> = LDSDL85::new(
        1,
        5,
        <usize>::MAX,
        DiscrepancyStrategy::Incremental,
        <usize>::MAX,
        2,
        Specialization::None,
        LowerBoundHeuristic::None,
        BranchingType::None,
        CacheInit::WithMemoryDynamic,
        0,
        true,
        heuristic.as_mut(),
    );
    algo.fit(&mut structure);

    println!("Statistics: {:?}", algo.statistics);
    algo.tree.print();
}
