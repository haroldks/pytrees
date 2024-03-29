use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Support};
use float_cmp::{ApproxEq, F64Margin};

pub type DataStructure<'a> = RSparseBitsetStructure<'a>;

pub trait Heuristic {
    fn compute(&self, structure: &mut DataStructure, candidates: &mut Vec<Attribute>);
}

#[derive(Default)]
pub struct NoHeuristic;

impl Heuristic for NoHeuristic {
    fn compute(&self, _structure: &mut DataStructure, _candidates: &mut Vec<Attribute>) {}
}

#[derive(Default)]
pub struct GiniIndex;

impl Heuristic for GiniIndex {
    fn compute(&self, structure: &mut DataStructure, candidates: &mut Vec<Attribute>) {
        let mut root_classes_support = structure.labels_support().to_vec();
        let mut candidates_sorted = vec![];
        for attribute in candidates.iter() {
            let gini = Self::gini_index(*attribute, structure, &root_classes_support);
            candidates_sorted.push((*attribute, gini));
        }
        candidates_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        *candidates = candidates_sorted
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<Attribute>>();
    }
}

impl GiniIndex {
    fn gini_index(
        attribute: Attribute,
        structure: &mut DataStructure,
        root_classes_support: &[usize],
    ) -> f64 {
        let _ = structure.push((attribute, 0));
        let left_classes_supports = structure.labels_support().to_vec();
        structure.backtrack();

        let right_classes_support = root_classes_support
            .iter()
            .enumerate()
            .map(|(idx, val)| *val - left_classes_supports[idx])
            .collect::<Vec<usize>>();

        let actual_size = root_classes_support.iter().sum::<usize>() as f64;
        let left_split_size = left_classes_supports.iter().sum::<usize>();
        let right_split_size = right_classes_support.iter().sum::<usize>();

        let mut left_gini_index = 0f64;
        let mut right_gini_index = 0f64;

        for class in 0..root_classes_support.len() {
            let p = match left_split_size {
                0 => 0f64,
                _ => (left_classes_supports[class] as f64 / left_split_size as f64).powf(2.),
            };

            left_gini_index += p;

            let p = match right_split_size {
                0 => 0f64,
                _ => (right_classes_support[class] as f64 / right_split_size as f64).powf(2.),
            };

            right_gini_index += p
        }
        ((left_split_size as f64) * (1. - left_gini_index)
            + (right_split_size as f64) * (1. - right_gini_index))
            / actual_size
    }
}

#[derive(Default)]
pub struct InformationGain;

impl Handler for InformationGain {}

impl Heuristic for InformationGain {
    fn compute(&self, structure: &mut DataStructure, candidates: &mut Vec<Attribute>) {
        self.internally_compute(structure, candidates, false);
    }
}

#[derive(Default)]
pub struct InformationGainRatio;

impl Handler for InformationGainRatio {}

impl Heuristic for InformationGainRatio {
    fn compute(&self, structure: &mut DataStructure, candidates: &mut Vec<Attribute>) {
        self.internally_compute(structure, candidates, true);
    }
}

// Information Gain and Information Gain Ratio handler

trait Handler {
    fn internally_compute(
        &self,
        structure: &mut DataStructure,
        attributes: &mut Vec<Attribute>,
        ratio: bool,
    ) {
        let root_classes_support = structure.labels_support().to_vec();
        let parent_entropy = Self::compute_entropy(&root_classes_support);
        let mut candidates_sorted = vec![];
        for attribute in attributes.iter() {
            let info_gain = Self::information_gain(
                *attribute,
                structure,
                &root_classes_support,
                parent_entropy,
                ratio,
            );
            candidates_sorted.push((*attribute, info_gain));
        }
        candidates_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        *attributes = candidates_sorted
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<Attribute>>();
    }

    fn information_gain(
        attribute: Attribute,
        structure: &mut DataStructure,
        root_classes_support: &[usize],
        parent_entropy: f64,
        ratio: bool,
    ) -> f64 {
        let _ = structure.push((attribute, 0));
        let left_classes_supports = structure.labels_support().to_vec();
        structure.backtrack();

        let right_classes_support = root_classes_support
            .iter()
            .enumerate()
            .map(|(idx, val)| *val - left_classes_supports[idx])
            .collect::<Vec<usize>>();

        let actual_size = root_classes_support.iter().sum::<usize>();
        let left_split_size = left_classes_supports.iter().sum::<usize>();
        let right_split_size = right_classes_support.iter().sum::<usize>();

        let left_weight = match actual_size {
            0 => 0f64,
            _ => left_split_size as f64 / actual_size as f64,
        };

        let right_weight = match actual_size {
            0 => 0f64,
            _ => right_split_size as f64 / actual_size as f64,
        };

        let mut split_info = 0f64;
        if ratio {
            if left_weight > 0. {
                split_info = -left_weight * left_weight.log2();
            }
            if right_weight > 0. {
                split_info += -right_weight * right_weight.log2();
            }
        }
        if split_info.approx_eq(
            0.,
            F64Margin {
                ulps: 2,
                epsilon: 0.0,
            },
        ) {
            split_info = 1f64;
        }

        let left_split_entropy = Self::compute_entropy(&left_classes_supports);
        let right_split_entropy = Self::compute_entropy(&right_classes_support);

        let info_gain = parent_entropy
            - (left_weight * left_split_entropy + right_weight * right_split_entropy);
        if ratio {
            return info_gain / split_info;
        }
        info_gain
    }

    fn compute_entropy(covers: &[usize]) -> f64 {
        let support = covers.iter().sum::<usize>();
        let mut entropy = 0f64;
        for class_support in covers {
            let p = match support {
                0 => 0f64,
                _ => *class_support as f64 / support as f64,
            };

            let mut log_val = 0f64;
            if p > 0. {
                log_val = p.log2();
            }
            entropy += -p * log_val;
        }
        entropy
    }
}
