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

mod algorithms;
mod dataset;
mod heuristics;
mod post_process;
mod structures;

fn main() {
    let dataset = BinaryDataset::load("test_data/hepatitis.txt", false, 0.0);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);

    let mut heuristic: Box<dyn Heuristic> = Box::new(NoHeuristic::default());

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

    let algo = LGDT::fit(&mut structure, 5, 2, InfoGain::fit);
    algo.print();
    // algo.fit(&mut structure);
    // algo.tree.print();
}
