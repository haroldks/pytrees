use paste::paste;
use pytrees::algorithms::algorithm_trait::{Algorithm, Basic};
use pytrees::algorithms::dl85::DL85;
use pytrees::algorithms::dl85_utils::structs_enums::{
    BranchingType, CacheInit, LowerBoundHeuristic, Specialization,
};
use pytrees::algorithms::idk::IDK;
use pytrees::algorithms::info_gain::InfoGain;
use pytrees::algorithms::lgdt::LGDT;
use pytrees::algorithms::murtree::MurTree;
use pytrees::dataset::binary_dataset::BinaryDataset;
use pytrees::dataset::data_trait::Dataset;
use pytrees::heuristics::{Heuristic, NoHeuristic};
use pytrees::structures::bitsets_structure::BitsetStructure;
use pytrees::structures::caching::trie::Data;
use pytrees::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use pytrees::structures::raw_binary_structure::RawBinaryStructure;
use pytrees::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use pytrees::structures::structure_trait::Structure;
use pytrees::structures::structures_types::{Depth, Support};

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

macro_rules! integration_tests_dl85 {
    ($($name:ident: $minsup:expr, $maxdepth:expr, $value:expr;)*) => {
        $(
            paste!{
                 #[test]
                 fn [<dl85_ $name _minsup_ $minsup _maxdepth_ $maxdepth >]() {
                    let data = BinaryDataset::load(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let sparse_data = RSparseBitsetStructure::format_input_data(&data);
                    let mut structure = RSparseBitsetStructure::new(&sparse_data);
                    for error in solve_instance_dl85(&mut structure, $minsup, $maxdepth) {
                        assert_eq!(error, $value);
                    }
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

fn solve_instance_dl85(
    structure: &mut RSparseBitsetStructure,
    minsup: Support,
    max_depth: Depth,
) -> Vec<usize> {
    let mut heuristic: Box<dyn Heuristic> = Box::new(NoHeuristic::default());
    let mut errors = vec![];

    let specializations = vec![Specialization::None, Specialization::Murtree];
    let lower_bounds = vec![LowerBoundHeuristic::None, LowerBoundHeuristic::Similarity];
    let branching_types = vec![BranchingType::None, BranchingType::Dynamic];

    for spec in specializations.iter() {
        for lower_bound in lower_bounds.iter() {
            for branching_type in branching_types.iter() {
                let mut algo: DL85<'_, _, Data> = DL85::new(
                    minsup,
                    max_depth,
                    <usize>::MAX,
                    100,
                    *spec,
                    *lower_bound,
                    *branching_type,
                    CacheInit::WithMemoryDynamic,
                    0,
                    true,
                    heuristic.as_mut(),
                );
                algo.fit(structure);
                errors.push(algo.statistics.tree_error);
                structure.reset();
            }
        }
    }

    errors
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

integration_tests_dl85! {
    anneal: 1, 2, 137;
    anneal: 1, 3, 112;
}
