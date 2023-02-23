use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::idk::IDK;
use crate::algorithms::info_gain::InfoGain;
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::pycore::less_greedy::FitMethod::Murtree;
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::horizontal_binary_structure::HorizontalBinaryStructure;
use crate::structures::raw_binary_structure::RawBinaryStructure;
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Depth, Support};
use numpy::PyReadonlyArrayDyn;
use pyo3::{pyclass, pymethods, pymodule, IntoPy, PyObject, PyResult, Python};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Serialize, Deserialize)]
enum DataStructure {
    RegularBitset,
    ReversibleSparseBitset,
    HorizontalData,
    RawBinaryData,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
enum FitMethod {
    InfoGain,
    Murtree,
}

#[derive(Clone, Copy)]
struct LGDTConstraints {
    min_sup: Support,
    max_depth: Depth,
    data_structure: DataStructure,
    method: FitMethod,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct LGDTStatistics {
    duration: Duration,
    error: usize,
}
impl IntoPy<PyObject> for LGDTStatistics {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let json = serde_json::to_string_pretty(&self).unwrap();
        json.into_py(py)
    }
}

#[pyclass]
pub(crate) struct LGDTInternalClassifier {
    tree: Tree<NodeData>,
    constraints: LGDTConstraints,
    statistics: LGDTStatistics,
}

#[pymethods]
impl LGDTInternalClassifier {
    #[new]
    fn new(min_sup: Support, max_depth: Depth, data_structure: &str, fit_method: &str) -> Self {
        let data_structure = match data_structure {
            "regular_bitset" => DataStructure::RegularBitset,
            "reversible_sparse_bitset" => DataStructure::ReversibleSparseBitset,
            "horizontal_data" => DataStructure::HorizontalData,
            "raw_binary_data" => DataStructure::RawBinaryData,
            _ => panic!("Invalid data structure"),
        };

        let method = match fit_method {
            "info_gain" => FitMethod::InfoGain,
            "murtree" => FitMethod::Murtree,
            _ => panic!("Invalid fit method"),
        };

        let constraints = LGDTConstraints {
            max_depth,
            min_sup,
            data_structure,
            method,
        };

        Self {
            tree: Tree::new(),
            constraints,
            statistics: LGDTStatistics {
                duration: Default::default(),
                error: 0,
            },
        }
    }

    fn train(&mut self, input: PyReadonlyArrayDyn<f64>, target: PyReadonlyArrayDyn<f64>) {
        let input = input.as_array().map(|a| *a as usize);
        let target = target.as_array().map(|a| *a as usize);
        let dataset = BinaryDataset::load_from_numpy(&input, &target);

        let output = match self.constraints.data_structure {
            DataStructure::RegularBitset => {
                let formatted_data = BitsetStructure::format_input_data(&dataset);
                let mut structure = BitsetStructure::new(&formatted_data);
                solve_instance(&mut structure, self.constraints.method, self.constraints)
            }
            DataStructure::ReversibleSparseBitset => {
                let formatted_data = RSparseBitsetStructure::format_input_data(&dataset);
                let mut structure = RSparseBitsetStructure::new(&formatted_data);
                solve_instance(&mut structure, self.constraints.method, self.constraints)
            }
            DataStructure::HorizontalData => {
                let formatted_data = HorizontalBinaryStructure::format_input_data(&dataset);
                let mut structure = HorizontalBinaryStructure::new(&formatted_data);
                solve_instance(&mut structure, self.constraints.method, self.constraints)
            }
            DataStructure::RawBinaryData => {
                let mut structure = RawBinaryStructure::new(&dataset);
                solve_instance(&mut structure, self.constraints.method, self.constraints)
            }
        };
        self.tree = output.0;
        self.statistics = output.1;
    }

    #[getter]
    fn statistics(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.statistics.into_py(py))
    }

    #[getter]
    fn tree(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.tree.clone().into_py(py))
    }
}

fn solve_instance<S: Structure>(
    structure: &mut S,
    method: FitMethod,
    constraints: LGDTConstraints,
) -> (Tree<NodeData>, LGDTStatistics) {
    let time = Instant::now();
    let tree = match method {
        FitMethod::InfoGain => {
            let method = InfoGain::fit;
            internal_solver(structure, method, &constraints)
        }
        FitMethod::Murtree => {
            let time = Instant::now();
            let method = MurTree::fit;
            internal_solver(structure, method, &constraints)
        }
    };
    let duration = time.elapsed();
    let error = LGDT::get_tree_error(&tree);
    (tree, LGDTStatistics { duration, error })
}

fn internal_solver<S, F>(
    structure: &mut S,
    fit_method: F,
    constraints: &LGDTConstraints,
) -> Tree<NodeData>
where
    S: Structure,
    F: Fn(&mut S, Support, Depth) -> Tree<NodeData>,
{
    match constraints.max_depth == 0 {
        true => IDK::fit(structure, constraints.min_sup, fit_method),
        false => LGDT::fit(
            structure,
            constraints.min_sup,
            constraints.max_depth,
            fit_method,
        ),
    }
}