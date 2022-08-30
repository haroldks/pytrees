#[macro_use]
extern crate bencher;

use bencher::Bencher;
use perf_lgdt::dataset::binary_dataset::BinaryDataset;
use perf_lgdt::dataset::data_trait::Dataset;
use perf_lgdt::structures::bitsets_structure::BitsetStructure;
use perf_lgdt::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use perf_lgdt::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use perf_lgdt::structures::structure_trait::Structure;

fn bench_create_horizontal_structure(b: &mut Bencher) {
    let datasets = [
        "datasets/small.txt",
        "datasets/small_.txt",
        "datasets/rsparse_dataset.txt",
        "datasets/anneal.txt",
        "datasets/splice-1.txt",
        "datasets/letter.txt",
        "datasets/ionosphere.txt",
    ];
    b.iter(|| {
        datasets.map(|file| {
            let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
            let bitset_data = HorizontalBinaryStructure::format_input_data(&dataset);
            HorizontalBinaryStructure::new(&bitset_data);
        })
    })
}

fn bench_create_simple_bitset_structure(b: &mut Bencher) {
    let datasets = [
        "datasets/small.txt",
        "datasets/small_.txt",
        "datasets/rsparse_dataset.txt",
        "datasets/anneal.txt",
        "datasets/splice-1.txt",
        "datasets/letter.txt",
        "datasets/ionosphere.txt",
    ];
    b.iter(|| {
        datasets.map(|file| {
            let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
            let bitset_data = BitsetStructure::format_input_data(&dataset);
            BitsetStructure::new(&bitset_data);
        })
    })
}

fn bench_create_rsparse_bitset_structure(b: &mut Bencher) {
    let datasets = [
        "datasets/small.txt",
        "datasets/small_.txt",
        "datasets/rsparse_dataset.txt",
        "datasets/anneal.txt",
        "datasets/splice-1.txt",
        "datasets/letter.txt",
        "datasets/ionosphere.txt",
    ];
    b.iter(|| {
        datasets.map(|file| {
            let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
            let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
            RSparseBitsetStructure::new(&bitset_data);
        })
    })
}

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
    benches,
    // bench_create_horizontal_structure,
    // bench_create_simple_bitset_structure,
    // bench_create_rsparse_bitset_structure,
    bench_running_through_using_data_using_horizontal,
    bench_running_through_using_data_using_simple_bitset,
    bench_running_through_using_data_using_sparse_bitset
);
benchmark_main!(benches);
