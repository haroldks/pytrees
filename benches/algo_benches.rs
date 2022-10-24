use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use perf_lgdt::algorithms::algorithm_trait::Algorithm;
use perf_lgdt::algorithms::info_gain::InfoGain;
use perf_lgdt::algorithms::lgdt::LGDT;
use perf_lgdt::algorithms::murtree::MurTree;
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

fn compare_struct_on_dataset(c: &mut Criterion) {
    let filename = "datasets/mushroom.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut rsparse_struct = RSparseBitsetStructure::new(&bitset_data);
    let mut bitset_struct = BitsetStructure::new(&bitset_data);
    let horizontal_data = HorizontalBinaryStructure::format_input_data(&dataset);
    let mut hz_struct = HorizontalBinaryStructure::new(&horizontal_data);
    let mut group = c.benchmark_group("LGDT");

    for depth in 1..11 {
        for minsup in [1, 5] {
            let parameter = (minsup, depth);
            let parameter_string = format!("min_d {} / depth {}", minsup, depth);
            group.bench_with_input(
                BenchmarkId::new("horizontal", &parameter_string),
                &parameter,
                |b, (depth, minsup)| {
                    b.iter(|| {
                        LGDT::fit(
                            &mut hz_struct,
                            black_box(*minsup),
                            black_box(*depth),
                            MurTree::fit,
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("bitset", &parameter_string),
                &(depth, minsup),
                |b, (depth, minsup)| {
                    b.iter(|| {
                        LGDT::fit(
                            &mut bitset_struct,
                            black_box(*minsup),
                            black_box(*depth),
                            MurTree::fit,
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("rsparse", &parameter_string),
                &(depth, minsup),
                |b, (depth, minsup)| {
                    b.iter(|| {
                        LGDT::fit(
                            &mut rsparse_struct,
                            black_box(*minsup),
                            black_box(*depth),
                            MurTree::fit,
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    compare_struct_on_anneal,
    //anneal_horiz_benchmark,
    //anneal_bitset_benchmark,
    //anneal_rsparse_benchmark
);
criterion_main!(benches);
