use pyo3::prelude::PyModule;
use pyo3::{pyclass, pymethods, pymodule, IntoPy, PyObject, PyResult, Python};

use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::{
    BranchingType, CacheInit, Constraints, DiscrepancyStrategy, LowerBoundHeuristic, SortHeuristic,
    Specialization, Statistics,
};
use crate::algorithms::lds_dl85::LDSDL85;
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
        discrepancy_budget: isize,
        discrepancy_strategy: usize,
        error: isize,
        time: isize,
        specialization: usize,
        lower_bound: usize,
        branching_type: usize,
        one_time_sort: bool,
        heuristic: usize,
        cache_init: usize,
        cache_init_size: usize,
    ) -> Self {
        let max_error = match error == -1 {
            true => <usize>::MAX,
            false => error as usize,
        };

        let max_time = match time == -1 {
            true => <usize>::MAX,
            false => time as usize,
        };

        let discrepancy_budget = match discrepancy_budget == -1 {
            true => <usize>::MAX,
            false => discrepancy_budget as usize,
        };

        let specialization = match specialization {
            0 => Specialization::None,
            1 => Specialization::Murtree,
            _ => panic!("Invalid specialization"),
        };

        let lower_bound = match lower_bound {
            0 => LowerBoundHeuristic::None,
            1 => LowerBoundHeuristic::Similarity,
            _ => panic!("Invalid lower bound"),
        };

        let heuristic = match heuristic {
            0 => SortHeuristic::None,
            1 => SortHeuristic::InformationGain,
            2 => SortHeuristic::InformationGainRatio,
            3 => SortHeuristic::GiniIndex,
            _ => panic!("Invalid heuristic"),
        };

        let branching = match branching_type {
            0 => BranchingType::None,
            1 => BranchingType::Dynamic,
            _ => panic!("Invalid branching type"),
        };

        let cache_init = match cache_init {
            0 => CacheInit::Normal,
            1 => CacheInit::WithMemoryDynamic,
            2 => CacheInit::WithMemoryFromUser,
            _ => panic!("Invalid cache init type"),
        };

        let discrepancy_strategy = match discrepancy_strategy {
            0 => DiscrepancyStrategy::None,
            1 => DiscrepancyStrategy::Incremental,
            2 => DiscrepancyStrategy::Double,
            _ => panic!("Invalid discrepancy strategy"),
        };

        let constraints = Constraints {
            max_depth,
            min_sup,
            max_error,
            max_time,
            one_time_sort,
            specialization,
            lower_bound,
            branching,
            cache_init,
            cache_init_size,
            discrepancy_budget,
            discrepancy_strategy,
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
            SortHeuristic::InformationGain => Box::<InformationGain>::default(),
            SortHeuristic::InformationGainRatio => Box::<InformationGainRatio>::default(),
            SortHeuristic::GiniIndex => Box::<GiniIndex>::default(),
            SortHeuristic::None => Box::<NoHeuristic>::default(),
        };

        if let DiscrepancyStrategy::None = self.constraints.discrepancy_strategy {
            let mut algorithm: DL85<'_, _, Data> = DL85::new(
                self.constraints.min_sup,
                self.constraints.max_depth,
                self.constraints.max_error,
                self.constraints.max_time,
                self.constraints.specialization,
                self.constraints.lower_bound,
                self.constraints.branching,
                self.constraints.cache_init,
                self.constraints.cache_init_size,
                self.constraints.one_time_sort,
                heuristic.as_mut(),
            );

            algorithm.fit(&mut structure);
            self.tree = algorithm.tree.clone();
            self.statistics = algorithm.statistics;
        } else {
            let mut algorithm: LDSDL85<'_, _, Data> = LDSDL85::new(
                self.constraints.min_sup,
                self.constraints.max_depth,
                self.constraints.discrepancy_budget,
                self.constraints.discrepancy_strategy,
                self.constraints.max_error,
                self.constraints.max_time,
                self.constraints.specialization,
                self.constraints.lower_bound,
                self.constraints.branching,
                self.constraints.cache_init,
                self.constraints.cache_init_size,
                self.constraints.one_time_sort,
                heuristic.as_mut(),
            );

            algorithm.fit(&mut structure);
            self.tree = algorithm.tree.clone();
            self.statistics = algorithm.statistics;
        }
    }
}
