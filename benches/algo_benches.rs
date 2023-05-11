use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use pytrees::algorithms::algorithm_trait::Algorithm;
use pytrees::algorithms::info_gain::InfoGain;
use pytrees::algorithms::lgdt::LGDT;
use pytrees::algorithms::murtree::MurTree;
use pytrees::dataset::binary_dataset::BinaryDataset;
use pytrees::dataset::data_trait::Dataset;
use pytrees::structures::bitsets_structure::BitsetStructure;
use pytrees::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use pytrees::structures::raw_binary_structure::RawBinaryStructure;
use pytrees::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use pytrees::structures::structure_trait::Structure;

pub fn anneal_horiz_benchmark(c: &mut Criterion) {
    let filename = "test_data/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = HorizontalBinaryStructure::format_input_data(&dataset);
    let mut structure = HorizontalBinaryStructure::new(&bitset_data);
    c.bench_function("hz_anneal", |b| {
        b.iter(|| LGDT::fit(&mut structure, 1, 5, InfoGain::fit))
    });
}

pub fn anneal_bitset_benchmark(c: &mut Criterion) {
    let filename = "test_data/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = BitsetStructure::format_input_data(&dataset);
    let mut structure = BitsetStructure::new(&bitset_data);
    c.bench_function("bit_anneal", |b| {
        b.iter(|| LGDT::fit(&mut structure, 1, 5, InfoGain::fit))
    });
}

pub fn anneal_rsparse_benchmark(c: &mut Criterion) {
    let filename = "test_data/anneal.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset_data);
    c.bench_function("rsparse_anneal", |b| {
        b.iter(|| LGDT::fit(&mut structure, 1, 5, InfoGain::fit))
    });
}

fn compare_struct_on_dataset(c: &mut Criterion) {
    let filename = "test_data/letter.txt";
    let dataset = BinaryDataset::load(filename, false, 0.0);
    let mut raw_struct = RawBinaryStructure::new(&dataset);
    let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    let mut rsparse_struct = RSparseBitsetStructure::new(&bitset_data);
    let mut bitset_struct = BitsetStructure::new(&bitset_data);
    let horizontal_data = HorizontalBinaryStructure::format_input_data(&dataset);
    let mut hz_struct = HorizontalBinaryStructure::new(&horizontal_data);
    let mut group = c.benchmark_group("LGDT");
    // group.sampling_mode(SamplingMode::Flat);

    for depth in [2, 3, 5, 7] {
        for minsup in [5] {
            let parameter = (minsup, depth);
            let parameter_string = format!("s_{}_d_{}", minsup, depth);

            group.bench_with_input(
                BenchmarkId::new("raw", &parameter_string),
                &parameter,
                |b, (minsup, depth)| {
                    b.iter(|| {
                        LGDT::fit(
                            &mut raw_struct,
                            black_box(*minsup),
                            black_box(*depth),
                            MurTree::fit,
                        )
                    })
                },
            );
            // group.bench_with_input(
            //     BenchmarkId::new("horizontal", &parameter_string),
            //     &parameter,
            //     |b, (minsup, depth)| {
            //         b.iter(|| {
            //             LGDT::fit(
            //                 &mut hz_struct,
            //                 black_box(*minsup),
            //                 black_box(*depth),
            //                 MurTree::fit,
            //             )
            //         })
            //     },
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("bitset", &parameter_string),
            //     &parameter,
            //     |b, (minsup, depth)| {
            //         b.iter(|| {
            //             LGDT::fit(
            //                 &mut bitset_struct,
            //                 black_box(*minsup),
            //                 black_box(*depth),
            //                 MurTree::fit,
            //             )
            //         })
            //     },
            // );
            group.bench_with_input(
                BenchmarkId::new("rsparse", &parameter_string),
                &parameter,
                |b, (minsup, depth)| {
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
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = compare_struct_on_dataset
    //anneal_horiz_benchmark,
    //anneal_bitset_benchmark,
    //anneal_rsparse_benchmark
);
criterion_main!(benches);
