use pyo3::prelude::PyModule;
use pyo3::{pyclass, pymethods, pymodule, IntoPy, PyObject, PyResult, Python};

use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::{
    BranchingType, Constraints, LowerBoundHeuristic, SortHeuristic, Specialization, Statistics,
};
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::caching::trie::Data;
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structures_types::{Depth, Support};
use numpy::PyReadonlyArrayDyn;
use std::time::Duration;

#[pyclass]
pub(crate) struct Dl85InternalClassifier {
    heuristic: SortHeuristic,
    tree: Tree<NodeData>,
    constraints: Constraints,
    statistics: Statistics,
}

#[pymethods]
impl Dl85InternalClassifier {
    #[new]
    fn new(
        min_sup: Support,
        max_depth: Depth,
        error: isize,
        time: isize,
        specialization: &str,
        lower_bound: &str,
        one_time_sort: bool,
        heuristic: &str,
    ) -> Self {
        let max_error = match error == -1 {
            true => <usize>::MAX,
            false => error as usize,
        };

        let max_time = match time == -1 {
            true => <usize>::MAX,
            false => error as usize,
        };

        let specialization = match specialization {
            "none" => Specialization::None,
            "murtree" => Specialization::Murtree,
            _ => panic!("Invalid specialization"),
        };

        let lower_bound = match lower_bound {
            "none" => LowerBoundHeuristic::None,
            "similarity" => LowerBoundHeuristic::Similarity,
            _ => panic!("Invalid lower bound"),
        };

        let heuristic = match heuristic {
            "info_gain" => SortHeuristic::InformationGain,
            "info_gain_ratio" => SortHeuristic::InformationGainRatio,
            "gini_index" => SortHeuristic::GiniIndex,
            "no_heuristic" => SortHeuristic::None,
            _ => panic!("Invalid heuristic"),
        };

        let constraints = Constraints {
            max_depth,
            min_sup,
            max_error,
            max_time,
            one_time_sort,
            specialization,
            lower_bound,
            branching: BranchingType::None,
        };

        let statistics = Statistics {
            num_attributes: 0,
            num_samples: 0,
            train_distribution: [0, 0],
            constraints,
            cache_size: 0,
            tree_error: 0,
            duration: Duration::default(),
        };

        Self {
            heuristic,
            tree: Tree::new(),
            constraints,
            statistics,
        }
    }

    #[getter]
    fn statistics(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.statistics.into_py(py))
    }

    #[getter]
    fn tree(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.tree.clone().into_py(py))
    }

    fn train(&mut self, input: PyReadonlyArrayDyn<f64>, target: PyReadonlyArrayDyn<f64>) {
        let input = input.as_array().map(|a| *a as usize);
        let target = target.as_array().map(|a| *a as usize);
        let dataset = BinaryDataset::load_from_numpy(&input, &target);
        let formatted_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&formatted_data);

        let mut heuristic: Box<dyn Heuristic> = match self.heuristic {
            SortHeuristic::InformationGain => Box::new(InformationGain::default()),
            SortHeuristic::InformationGainRatio => Box::new(InformationGainRatio::default()),
            SortHeuristic::GiniIndex => Box::new(GiniIndex::default()),
            SortHeuristic::None => Box::new(NoHeuristic::default()),
        };

        let mut algorithm: DL85<'_, _, Data> = DL85::new(
            self.constraints.min_sup,
            self.constraints.max_depth,
            self.constraints.max_error,
            self.constraints.max_error,
            self.constraints.specialization,
            self.constraints.lower_bound,
            BranchingType::Dynamic,
            self.constraints.one_time_sort,
            heuristic.as_mut(),
        );

        algorithm.fit(&mut structure);
        self.tree = algorithm.tree.clone();
        self.statistics = algorithm.statistics;
    }
}
