// #![allow(unused)]
#![warn(clippy::too_many_arguments)]
use crate::algorithms::algorithm_trait::Algorithm;

use crate::dataset::data_trait::Dataset;

use crate::structures::structure_trait::Structure;

use crate::heuristics::Heuristic;
use crate::pycore::less_greedy::{LGDTInternalClassifier, ParallelLGDTInternalClassifier};
use crate::pycore::optimal::Dl85InternalClassifier;

use pyo3::prelude::PyModule;
use pyo3::{pymodule, IntoPy, PyResult, Python};

extern crate core;
pub mod algorithms;
pub mod dataset;
pub mod heuristics;
mod post_process;
mod pycore;
pub mod structures;

#[pymodule]
fn pytrees_internal(py: Python<'_>, _m: &PyModule) -> PyResult<()> {
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
