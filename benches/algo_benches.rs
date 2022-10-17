use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perf_lgdt::algorithms::algorithm_trait::Algorithm;
use perf_lgdt::algorithms::info_gain::InfoGain;
use perf_lgdt::algorithms::lgdt::LGDT;
use perf_lgdt::dataset::binary_dataset::BinaryDataset;
use perf_lgdt::dataset::data_trait::Dataset;
use perf_lgdt::structures::bitsets_structure::BitsetStructure;
use perf_lgdt::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use perf_lgdt::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use perf_lgdt::structures::structure_trait::Structure;

pub fn anneal_horiz_benchmark(c: &mut Criterion) {
    let filename = "datasets/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = HorizontalBinaryStructure::format_input_data(&dataset);
    let mut structure = HorizontalBinaryStructure::new(&bitset_data);
    c.bench_function("hz_anneal", |b| {
        b.iter(|| LGDT::fit(&mut structure, 1, 5, InfoGain::fit))
    });
}

pub fn anneal_bitset_benchmark(c: &mut Criterion) {
    let filename = "datasets/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = BitsetStructure::format_input_data(&dataset);
    let mut structure = BitsetStructure::new(&bitset_data);
    c.bench_function("bit_anneal", |b| {
        b.iter(|| LGDT::fit(&mut structure, 1, 5, InfoGain::fit))
    });
}

pub fn anneal_rsparse_benchmark(c: &mut Criterion) {
    let filename = "datasets/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);
    c.bench_function("rsparse_anneal", |b| {
        b.iter(|| LGDT::fit(&mut structure, 1, 5, InfoGain::fit))
    });
}

criterion_group!(
    benches,
    anneal_horiz_benchmark,
    anneal_bitset_benchmark,
    anneal_rsparse_benchmark
);
criterion_main!(benches);
