use paste::paste;
use perf_lgdt::algorithms::algorithm_trait::{Algorithm, Basic};
use perf_lgdt::algorithms::idk::IDK;
use perf_lgdt::algorithms::info_gain::InfoGain;
use perf_lgdt::algorithms::lgdt::LGDT;
use perf_lgdt::algorithms::murtree::MurTree;
use perf_lgdt::dataset::binary_dataset::BinaryDataset;
use perf_lgdt::dataset::data_trait::Dataset;
use perf_lgdt::structures::bitsets_structure::BitsetStructure;
use perf_lgdt::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use perf_lgdt::structures::raw_binary_structure::RawBinaryStructure;
use perf_lgdt::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use perf_lgdt::structures::structure_trait::Structure;
use perf_lgdt::structures::structures_types::{Depth, Support};

macro_rules! integration_tests_lgdt {
    ($($name:ident: $minsup:expr, $depth:expr, $algo:expr, $value:expr;)*) => {
        $(
            paste!{

                 #[test]
                 fn [<lgdt_ $name _ $algo _raw_ $name _minsup_ $minsup _depth_ $depth>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let mut structure = RawBinaryStructure::new(&data);
                    assert_eq!(solve_instance_lgdt(&mut structure, $minsup, $depth, $algo), $value);
                }

                #[test]
                 fn [<lgdt_ $name _ $algo _horizontal_ $name _minsup_ $minsup _depth_ $depth>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let horizontal_data = HorizontalBinaryStructure::format_input_data(&data);
                    let mut structure = HorizontalBinaryStructure::new(&horizontal_data);
                    assert_eq!(solve_instance_lgdt(&mut structure, $minsup, $depth, $algo), $value);
                }


                #[test]
                 fn [<lgdt_ $name _ $algo _bitset_ $name _minsup_ $minsup _depth_ $depth>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let horizontal_data = BitsetStructure::format_input_data(&data);
                    let mut structure = BitsetStructure::new(&horizontal_data);
                    assert_eq!(solve_instance_lgdt(&mut structure, $minsup, $depth, $algo), $value);
                }

                #[test]
                 fn [<lgdt_ $name _ $algo _rsparse_ $name _minsup_ $minsup _depth_ $depth>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let horizontal_data = RSparseBitsetStructure::format_input_data(&data);
                    let mut structure = RSparseBitsetStructure::new(&horizontal_data);
                    assert_eq!(solve_instance_lgdt(&mut structure, $minsup, $depth, $algo), $value);
                }
            }
        )*
    }
}

macro_rules! integration_tests_idk {
    ($($name:ident: $minsup:expr, $algo:expr, $value:expr;)*) => {
        $(
            paste!{
                 #[test]
                 fn [<idk_ $name _ $algo _raw_ $name _minsup_ $minsup>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let mut structure = RawBinaryStructure::new(&data);
                    assert_eq!(solve_instance_idk(&mut structure, $minsup, $algo), $value);
                }

                #[test]
                 fn [<idk_ $name _ $algo _horizontal_ $name _minsup_ $minsup>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let horizontal_data = HorizontalBinaryStructure::format_input_data(&data);
                    let mut structure = HorizontalBinaryStructure::new(&horizontal_data);
                    assert_eq!(solve_instance_idk(&mut structure, $minsup, $algo), $value);
                }


                #[test]
                 fn [<idk_ $name _ $algo _bitset_ $name _minsup_ $minsup>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let horizontal_data = BitsetStructure::format_input_data(&data);
                    let mut structure = BitsetStructure::new(&horizontal_data);
                    assert_eq!(solve_instance_idk(&mut structure, $minsup, $algo), $value);
                }

                #[test]
                 fn [<idk_ $name _ $algo _rsparse_ $name _minsup_ $minsup>]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let horizontal_data = RSparseBitsetStructure::format_input_data(&data);
                    let mut structure = RSparseBitsetStructure::new(&horizontal_data);
                    assert_eq!(solve_instance_idk(&mut structure, $minsup, $algo), $value);
                }
            }
        )*
    }
}

fn solve_instance_lgdt<S>(structure: &mut S, minsup: Support, depth: Depth, algo: &str) -> usize
where
    S: Structure,
{
    let tree = match algo.eq("murtree") {
        true => LGDT::fit(structure, minsup, depth, MurTree::fit),
        false => LGDT::fit(structure, minsup, depth, InfoGain::fit),
    };
    LGDT::get_tree_error(&tree)
}

fn solve_instance_idk<S>(structure: &mut S, minsup: Support, algo: &str) -> usize
where
    S: Structure,
{
    let tree = match algo.eq("murtree") {
        true => IDK::fit(structure, minsup, MurTree::fit),
        false => IDK::fit(structure, minsup, InfoGain::fit),
    };
    LGDT::get_tree_error(&tree)
}

integration_tests_lgdt! {
    mushroom: 100, 4, "murtree", 108;
    mushroom: 1, 1, "murtree", 920;
    mushroom: 1, 1, "infogain", 920;
    mushroom: 5, 1, "murtree", 920;
    mushroom: 5, 1, "infogain", 920;
    mushroom: 1, 2, "murtree", 252;
    mushroom: 1, 2, "infogain", 396;
    mushroom: 5, 2, "murtree", 252;
    mushroom: 5, 2, "infogain", 396;
    mushroom: 1, 3, "murtree", 76;
    mushroom: 1, 3, "infogain", 76;
    mushroom: 5, 3, "murtree", 76;
    mushroom: 5, 3, "infogain", 76;
    mushroom: 1, 4, "murtree", 0;
    mushroom: 1, 4, "infogain", 0;
    mushroom: 5, 4, "murtree", 0;
    mushroom: 5, 4, "infogain", 0;
}

integration_tests_idk! {
    anneal: 1, "murtree", 34;
    mushroom: 1, "infogain", 0;
    mushroom: 1, "murtree", 0;
}
