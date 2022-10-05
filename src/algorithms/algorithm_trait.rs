use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Support, TreeIndex};

pub(crate) trait Algorithm {
    fn fit<S>(structure: &mut S, min_sup: Support, max_depth: Depth) -> Tree<NodeData<usize>>
    where
        S: Structure;

    fn first_candidates<S>(structure: &mut S, min_sup: Support) -> Vec<Attribute>
    where
        S: Structure,
    {
        let num_attributes = structure.num_attributes();
        let mut candidates = vec![];
        for i in 0..num_attributes {
            if structure.temp_push((i, 0)) >= min_sup && structure.temp_push((i, 1)) >= min_sup {
                candidates.push(i);
            }
        }
        candidates
    }

    fn sort_candidates<S, F>(
        structure: &mut S,
        candidates: &Vec<Attribute>,
        func: F,
        increasing: bool,
    ) -> Vec<Attribute>
    where
        S: Structure,
        F: Fn(&mut S, &Vec<Attribute>, bool) -> Vec<Attribute>,
    {
        func(structure, candidates, increasing)
    }

    fn get_tree_error(tree: &Tree<NodeData<usize>>) -> usize {
        if let Some(root) = tree.get_node(tree.get_root_index()) {
            return root.value.metric;
        }
        <usize>::MAX
    }

    fn get_misclassification_error(classes_support: &Vec<usize>) -> usize {
        classes_support.iter().sum::<usize>() - classes_support.iter().max().unwrap()
    }

    fn get_top_class(classes_support: &Vec<usize>) -> usize {
        classes_support
            .iter()
            .enumerate()
            .fold((0, classes_support[0]), |(idxm, valm), (idx, val)| {
                if val > &valm {
                    (idx, *val)
                } else {
                    (idxm, valm)
                }
            })
            .0
    }

    fn build_tree_recurse(tree: &mut Tree<NodeData<usize>>, parent: TreeIndex, depth: Depth) {
        if depth == 0 {
            if let Some(parent_node) = tree.get_node_mut(parent) {
                parent_node.left = 0;
                parent_node.right = 0;
            }
        } else {
            let left = tree.add_node(
                parent,
                true,
                TreeNode {
                    value: NodeData {
                        test: 0,
                        metric: <usize>::MAX,
                        out: None,
                    },
                    index: 0,
                    left: 0,
                    right: 0,
                },
            );
            Self::build_tree_recurse(tree, left, depth - 1);

            let right = tree.add_node(
                parent,
                false,
                TreeNode {
                    value: NodeData {
                        test: 0,
                        metric: <usize>::MAX,
                        out: None,
                    },
                    index: 0,
                    left: 0,
                    right: 0,
                },
            );
            Self::build_tree_recurse(tree, right, depth - 1);
        }
    }
}
