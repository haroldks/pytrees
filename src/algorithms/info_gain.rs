use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::murtree::MurTree;
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Item, Support};
use float_cmp::{ApproxEq, F64Margin};

pub struct InfoGain {
    tree: Option<Tree<NodeData>>,
    error: usize,
}

impl Basic for InfoGain {}

impl Algorithm for InfoGain {
    fn build_depth_one_tree<S>(structure: &mut S, min_sup: Support) -> Tree<NodeData>
    where
        S: Structure,
    {
        let mut candidates = Self::generate_candidates_list(structure, min_sup);

        if candidates.is_empty() {
            return Self::empty_tree(1);
        }

        let mut left_index = 0;
        let mut right_index = 0;

        let info_gain = InfoGain::sort_using_information_gain(structure, &mut candidates, false);
        let mut tree = Self::empty_tree(1);

        if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
            root.value.test = Some(candidates[0]);
            left_index = root.left;
            right_index = root.right;
        }

        let mut node_error = 0;

        for branch in [0usize, 1].iter() {
            structure.push((candidates[0], *branch));
            let classes_support = structure.labels_support();
            let error = Self::get_leaf_error(classes_support);

            let index = match *branch == 0 {
                true => left_index,
                false => right_index,
            };

            if let Some(node) = tree.get_node_mut(index) {
                node.value.error = error.0;
                node.value.out = Some(error.1);
                node_error += error.0;
            }
            structure.backtrack();
        }

        if let Some(root) = tree.get_node_mut(tree.get_root_index()) {
            root.value.error = node_error;
            root.value.metric = Some(info_gain);
        }
        tree
    }

    fn build_depth_two_tree<S>(structure: &mut S, min_sup: Support) -> Tree<NodeData>
    where
        S: Structure,
    {
        let candidates = InfoGain::generate_candidates_list(structure, min_sup);
        if candidates.is_empty() {
            return Self::empty_tree(2);
        }
        if candidates.len() < 2 {
            return Self::build_depth_one_tree(structure, min_sup);
        }

        let matrix = MurTree::build_depth_two_matrix(structure, &candidates);

        let classes_support = structure.labels_support();
        let support = classes_support.iter().sum::<usize>();

        let parent_entropy = Self::compute_entropy(classes_support);

        if parent_entropy.approx_eq(
            0.,
            F64Margin {
                ulps: 2,
                epsilon: 0.0,
            },
        ) {
            return InfoGain::build_depth_one_tree(structure, min_sup);
        }

        let mut tree = Self::empty_tree(2);

        for (i, first) in candidates.iter().enumerate().take(candidates.len()) {
            let mut root_tree = Self::empty_tree(2);
            let mut left_index = 0;
            let mut right_index = 0;

            if let Some(root_node) = root_tree.get_node_mut(root_tree.get_root_index()) {
                root_node.value.test = Some(*first);
                left_index = root_node.left;
                right_index = root_node.right;
            }

            for (j, second) in candidates.iter().enumerate() {
                if i == j {
                    continue;
                }

                let mut left_leaves = vec![];
                let mut right_leaves = vec![];
                let mut root_error = 0;
                let mut root_gain = 0f64;

                for val in [0usize, 1].iter() {
                    let mut left_leaf_index = 0;
                    let mut right_leaf_index = 0;
                    let mut node_gain = parent_entropy;

                    left_leaves = Self::get_leaves_classes_support(
                        &matrix,
                        (i, *val),
                        (j, 0),
                        classes_support,
                    );

                    let weight = match support {
                        0 => 0f64,
                        _ => left_leaves.iter().sum::<usize>() as f64 / support as f64,
                    };

                    node_gain -= Self::compute_entropy(&left_leaves) * weight;
                    let left_leaves_error = Self::get_leaf_error(&left_leaves);

                    right_leaves = Self::get_leaves_classes_support(
                        &matrix,
                        (i, *val),
                        (j, 1),
                        classes_support,
                    );

                    let weight = match support {
                        0 => 0f64,
                        _ => right_leaves.iter().sum::<usize>() as f64 / support as f64,
                    };

                    node_gain -= Self::compute_entropy(&right_leaves) * weight;
                    let right_leaves_error = Self::get_leaf_error(&right_leaves);

                    let node_error = left_leaves_error.0 + right_leaves_error.0;

                    let mut past_info_gain = 0f64;
                    let index = match *val == 0 {
                        true => left_index,
                        false => right_index,
                    };
                    if let Some(node) = root_tree.get_node(index) {
                        if node.value.metric.is_some() {
                            past_info_gain = node.value.metric.unwrap();
                        }
                    }

                    if node_gain > past_info_gain {
                        if let Some(node) = root_tree.get_node_mut(index) {
                            node.value.test = Some(*second);
                            node.value.error = node_error;
                            node.value.metric = Some(node_gain);
                            left_leaf_index = node.left;
                            right_leaf_index = node.right;
                        }
                        if let Some(left_leaf_ref) = root_tree.get_node_mut(left_leaf_index) {
                            left_leaf_ref.value.error = left_leaves_error.0;
                            left_leaf_ref.value.out = Some(left_leaves_error.1);

                            // Self::create_leaves(left_leaf_ref, &left_leaves, left_leaves_error);
                        }

                        if let Some(right_leaf_ref) = root_tree.get_node_mut(right_leaf_index) {
                            right_leaf_ref.value.error = right_leaves_error.0;
                            right_leaf_ref.value.out = Some(right_leaves_error.1);
                            // Self::create_leaves(right_leaf_ref, &right_leaves, right_leaves_error);
                        }
                    }
                    if let Some(node) = root_tree.get_node_mut(index) {
                        root_error += node.value.error;
                        root_gain += node.value.metric.unwrap();
                    }
                }
                if let Some(root_node) = root_tree.get_node_mut(root_tree.get_root_index()) {
                    root_node.value.error = root_error;
                    root_node.value.metric = Some(root_gain);
                    if root_node.value.error == 0 {
                        break;
                    }
                }
            }

            if Self::get_info_gain(&root_tree) > Self::get_info_gain(&tree) {
                tree = root_tree;
            }
            if Self::get_tree_error(&tree) == 0 {
                break;
            }
        }
        tree
    }
}

impl InfoGain {
    fn sort_using_information_gain<S>(
        structure: &mut S,
        candidates: &mut Vec<Attribute>,
        ratio: bool,
    ) -> f64
    where
        S: Structure,
    {
        let mut root_classes_support = structure.labels_support().to_vec();

        let parent_entropy = Self::compute_entropy(&root_classes_support);
        let mut candidates_sorted = vec![];
        for attribute in candidates.iter() {
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
        *candidates = candidates_sorted
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<Attribute>>();
        candidates_sorted[0].1
    }

    fn information_gain<S>(
        attribute: Attribute,
        structure: &mut S,
        root_classes_support: &[usize],
        parent_entropy: f64,
        ratio: bool,
    ) -> f64
    where
        S: Structure,
    {
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

    fn get_info_gain(tree: &Tree<NodeData>) -> f64 {
        if let Some(node) = tree.get_node(tree.get_root_index()) {
            if let Some(metric) = node.value.metric {
                return metric;
            }
        }
        0.
    }

    fn get_leaves_classes_support(
        matrix: &[Vec<Vec<usize>>],
        first: Item,
        second: Item,
        root_classes_support: &[usize],
    ) -> Vec<usize> {
        let i = first.0;
        let j = second.0;
        let is_left_i = first.1 == 0;
        let is_left_j = second.1 == 0;

        let i_right_sc = &matrix[i][i];
        let j_right_sc = &matrix[j][j];
        let i_right_j_right_sc = &matrix[i][j];

        let i_left_j_right_sc = Self::get_diff_errors(j_right_sc, i_right_j_right_sc);

        match is_left_i {
            true => {
                match is_left_j {
                    true => {
                        // i_left_j_left
                        let i_left_sc = Self::get_diff_errors(root_classes_support, i_right_sc);
                        Self::get_diff_errors(&i_left_sc, &i_left_j_right_sc) // i_left_j_left_sc
                    }

                    false => {
                        // i_left_j_right
                        i_left_j_right_sc.to_vec()
                    }
                }
            }
            false => {
                match is_left_j {
                    true => {
                        // i_right_j_left
                        Self::get_diff_errors(i_right_sc, i_right_j_right_sc) // i_right_j_left_sc
                    }

                    false => {
                        // i_right_j_right
                        i_right_j_right_sc.to_vec()
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod info_gain_test {
    use crate::algorithms::algorithm_trait::{Algorithm, Basic};
    use crate::algorithms::info_gain::InfoGain;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;

    #[test]
    fn gen_anneal_tree() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);
        let tree = InfoGain::build_depth_one_tree(&mut structure, 1);
        assert_eq!(InfoGain::get_tree_error(&tree), 152);
    }
}
