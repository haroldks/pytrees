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
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Statistics {
    pub(crate) constraints: Constraints,
    pub(crate) cache_size: usize,
    pub(crate) tree_error: usize,
    pub(crate) duration: Duration,
}

// End: Structures used in the algorithm

// Start: Enums used in the algorithm

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

enum Specialization {
    Murtree,
    Infogain,
    None,
}

// End: Enums used in the algorithm
