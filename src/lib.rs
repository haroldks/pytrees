#![allow(unused)]
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
use std::time::{Duration, Instant};

use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::{
    Constraints, LowerBoundHeuristic, SortHeuristic, Specialization, Statistics,
};
use crate::algorithms::idk::IDK;
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::pycore::less_greedy::{LGDTInternalClassifier, ParallelLGDTInternalClassifier};
use crate::pycore::optimal::Dl85InternalClassifier;
use crate::structures::caching::trie::Data;
use crate::structures::raw_binary_structure::RawBinaryStructure;
use log::info;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::PyModule;
use pyo3::{pyclass, pymodule, wrap_pymodule, IntoPy, PyObject, PyResult, Python};

extern crate core;
pub mod algorithms;
pub mod dataset;
pub mod heuristics;
mod post_process;
mod pycore;
pub mod structures;

#[pymodule]
fn pytrees_internal(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let optimal_module = pyo3::wrap_pymodule!(optimal);
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytrees_internal.optimal", optimal_module(py))?;

    let lgdt_module = pyo3::wrap_pymodule!(lgdt);
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytrees_internal.lgdt", lgdt_module(py))?;
    Ok(())
}

#[pymodule]
fn optimal(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<Dl85InternalClassifier>()?;
    Ok(())
}

#[pymodule]
fn lgdt(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<LGDTInternalClassifier>()?;
    module.add_class::<ParallelLGDTInternalClassifier>()?;
    Ok(())
}
