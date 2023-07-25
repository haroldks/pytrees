use pytrees::algorithms::algorithm_trait::{Algorithm, Basic};
use pytrees::algorithms::lgdt::LGDT;
use pytrees::algorithms::murtree::MurTree;
use pytrees::dataset::binary_dataset::BinaryDataset;
use pytrees::dataset::data_trait::Dataset;
use pytrees::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use pytrees::structures::structure_trait::Structure;
use std::time::Instant;

fn main() {
    let dataset = BinaryDataset::load("test_data/ionosphere.txt", false, 0.0);
    let bitset = RSparseBitsetStructure::format_input_data(&dataset);
    let mut runtimes = vec![];
    let mut errors = vec![];
    let num_threads = 5;
    for num_thread in 1..num_threads {
        println!("Num threads: {}", num_thread);
        let mut structure = RSparseBitsetStructure::new(&bitset, num_thread);
        let start = Instant::now();
        let tree = LGDT::fit(&mut structure, 1, 2, MurTree::fit);
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        runtimes.push(duration);
        let error = LGDT::get_tree_error(&tree);
        errors.push(error);
    }
    println!("Runtimes: {:?}", runtimes);
    println!("Errors: {:?}", errors);
}
