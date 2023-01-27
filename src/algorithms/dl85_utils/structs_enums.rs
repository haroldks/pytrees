use crate::structures::structures_types::{Depth, Support};
use std::time::Duration;

// Start: Structures used in the algorithm

#[derive(Debug, Clone, Copy)]
pub(crate) struct Constraints {
    pub max_depth: Depth,
    pub min_sup: Support,
    pub max_error: usize,
    pub max_time: usize,
    pub one_time_sort: bool,
    pub specialization: Specialization,
}

#[derive(Debug, Clone)]
pub struct Statistics {
    pub(crate) cache_size: usize,
    pub(crate) tree_error: usize,
    pub(crate) duration: Duration,
    pub(crate) num_attributes: usize,
    pub(crate) num_samples: usize,
    pub(crate) train_distribution: Vec<usize>,
    pub(crate) constraints: Constraints,
}

// End: Structures used in the algorithm

// Start: Enums used in the algorithm
#[derive(Debug, Clone, Copy)]
enum SortHeuristic {
    InformationGain,
    InformationGainRatio,
    GiniIndex,
    None,
}

// TODO: Not for here
enum LowerBoundHeuristic {
    Similarity,
    Simple,
    None,
}
#[derive(Debug, Clone, Copy)]
pub enum Specialization {
    Murtree,
    None,
}

// End: Enums used in the algorithm
