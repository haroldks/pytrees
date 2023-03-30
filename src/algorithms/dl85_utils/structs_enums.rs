use crate::structures::structures_types::{Depth, Support};
use pyo3::{IntoPy, PyObject, Python};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Start: Structures used in the algorithm

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Constraints {
    pub max_depth: Depth,
    pub min_sup: Support,
    pub max_error: usize,
    pub max_time: usize,
    pub one_time_sort: bool,
    pub specialization: Specialization,
    pub lower_bound: LowerBoundHeuristic,
    pub branching: BranchingType,
    pub cache_init: CacheInit,
    pub cache_init_size: usize,
    pub discrepancy_budget: usize,
    pub discrepancy_strategy: DiscrepancyStrategy,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Branching {
    pub branch: usize,
    pub lower_bound: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Statistics {
    pub(crate) cache_size: usize,
    pub tree_error: usize,
    pub(crate) duration: Duration,
    pub(crate) num_attributes: usize,
    pub(crate) num_samples: usize,
    pub(crate) train_distribution: [usize; 2],
    pub(crate) constraints: Constraints,
}

impl IntoPy<PyObject> for Statistics {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let json = serde_json::to_string_pretty(&self).unwrap();
        json.into_py(py)
    }
}

// End: Structures used in the algorithm

// Start: Enums used in the algorithm
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SortHeuristic {
    InformationGain,
    InformationGainRatio,
    GiniIndex,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LowerBoundHeuristic {
    Similarity,
    None,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Specialization {
    Murtree,
    None,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HasIntersected {
    Yes,
    No,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReturnCondition {
    Done,
    TimeLimitReached,
    LowerBoundConstrained,
    MaxDepthReached,
    NotEnoughSupport,
    PureNode,
    FromSpecializedAlgorithm,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BranchingType {
    Dynamic,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheInit {
    Normal,
    WithMemoryDynamic,
    WithMemoryFromUser,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DiscrepancyStrategy {
    None,
    Incremental,
    Double,
}

// End: Enums used in the algorithm
