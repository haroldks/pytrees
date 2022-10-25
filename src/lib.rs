use crate::algorithms::algorithm_trait::Algorithm;
use crate::algorithms::info_gain::InfoGain;
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Depth, Support};
use std::time::Instant;

use crate::algorithms::idk::IDK;
use crate::structures::raw_binary_structure::RawBinaryStructure;
use log::info;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, IntoPy, PyObject, PyResult, Python};

extern crate core;
pub mod algorithms;
pub mod dataset;
mod main;
mod post_process;
pub mod structures;

#[pymodule]
fn perf_lgdt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::try_init().expect("Logger already in use.");
    #[pyfn(m)]
    fn run<'py>(
        py: Python<'py>,
        input: PyReadonlyArrayDyn<f64>,
        target: PyReadonlyArrayDyn<f64>,
        min_sup: Support,
        max_depth: Depth,
        data_structure: &str,
        fit_method: &str,
        verbose: bool,
    ) -> PyObject {
        let input = input.as_array().map(|a| *a as usize);
        let target = target.as_array().map(|a| *a as usize);
        let dataset = BinaryDataset::load_from_numpy(&input, &target);

        if verbose && max_depth == 0 {
            info!("Use ID3 procedure to generate a full tree.");
        }

        let output = match data_structure {
            "bitset" => {
                if verbose {
                    info!("Using bitset data structure.");
                }

                let formatted_data = BitsetStructure::format_input_data(&dataset);
                let mut structure = BitsetStructure::new(&formatted_data);
                solve_instance(&mut structure, min_sup, max_depth, fit_method, verbose)
            }
            "sparse_bitset" => {
                if verbose {
                    info!("Using Reversible sparse bitset data structure.");
                }
                let formatted_data = RSparseBitsetStructure::format_input_data(&dataset);
                let mut structure = RSparseBitsetStructure::new(&formatted_data);
                solve_instance(&mut structure, min_sup, max_depth, fit_method, verbose)
            }

            "horizontal" => {
                if verbose {
                    info!("Using Reversible sparse bitset data structure.");
                }
                let formatted_data = RSparseBitsetStructure::format_input_data(&dataset);
                let mut structure = RSparseBitsetStructure::new(&formatted_data);
                solve_instance(&mut structure, min_sup, max_depth, fit_method, verbose)
            }

            _ => {
                if verbose {
                    info!("Using raw binary data structure.");
                }
                let mut structure = RawBinaryStructure::new(&dataset);
                solve_instance(&mut structure, min_sup, max_depth, fit_method, verbose)
            }
        };

        let json = serde_json::to_string_pretty(&output).unwrap();
        json.into_py(py)
    }

    Ok(())
}

fn solve_instance<S: Structure>(
    structure: &mut S,
    min_sup: Support,
    max_depth: Depth,
    fit_method: &str,
    verbose: bool,
) -> Tree<NodeData> {
    match fit_method {
        "infogain" => {
            if verbose {
                info!("Using Information Gain lookahead.");
            }
            let time = Instant::now();
            let tree = match max_depth == 0 {
                true => IDK::fit(structure, min_sup, InfoGain::fit),
                false => LGDT::fit(structure, min_sup, max_depth, InfoGain::fit),
            };
            let duration = time.elapsed();
            if verbose {
                info!("Run for {} milliseconds.", duration.as_millis());
            }
            tree
        }
        _ => {
            if verbose {
                info!("Using Murtree Depth 2 lookahead.");
            }
            let time = Instant::now();
            let tree = match max_depth == 0 {
                true => IDK::fit(structure, min_sup, MurTree::fit),
                false => LGDT::fit(structure, min_sup, max_depth, MurTree::fit),
            };
            let duration = time.elapsed();
            if verbose {
                info!("Run for {} milliseconds.", duration.as_millis());
            }
            tree
        }
    }
}
