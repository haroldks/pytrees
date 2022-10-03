#[macro_use]
extern crate bencher;

use bencher::Bencher;
use perf_lgdt::dataset::binary_dataset::BinaryDataset;
use perf_lgdt::dataset::data_trait::Dataset;
use perf_lgdt::structures::bitsets_structure::BitsetStructure;
use perf_lgdt::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use perf_lgdt::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use perf_lgdt::structures::structure_trait::Structure;

fn bench_running_through_using_data_using_horizontal(b: &mut Bencher) {
    let filename = "datasets/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let num_features = dataset.num_attributes();
    let bitset_data = HorizontalBinaryStructure::format_input_data(&dataset);
    let mut structure = HorizontalBinaryStructure::new(&bitset_data);

    b.iter(|| {
        for x in 0..num_features {
            let s = structure.push((x, 0));
            structure.backtrack();
        }
    })
}

fn bench_running_through_using_data_using_simple_bitset(b: &mut Bencher) {
    let filename = "datasets/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let num_features = dataset.num_attributes();
    let bitset_data = BitsetStructure::format_input_data(&dataset);
    let mut structure = BitsetStructure::new(&bitset_data);

    b.iter(|| {
        for x in 0..num_features {
            let s = structure.push((x, 0));
            structure.backtrack();
        }
    })
}

fn bench_running_through_using_data_using_sparse_bitset(b: &mut Bencher) {
    let filename = "datasets/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let num_features = dataset.num_attributes();
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);

    b.iter(|| {
        for x in 0..num_features {
            let s = structure.push((x, 0));
            structure.backtrack();
        }
    })
}

benchmark_group!(
    running_through_data_benches,
    bench_running_through_using_data_using_horizontal,
    bench_running_through_using_data_using_simple_bitset,
    bench_running_through_using_data_using_sparse_bitset
);

benchmark_main!(running_through_data_benches);
