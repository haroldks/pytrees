use crate::algorithms::algorithm_trait::Basic;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Support, TreeIndex};
use num_traits::Bounded;

struct LGDT {
    tree: Option<Tree<NodeData<usize>>>,
    error: Option<usize>,
}

impl Basic for LGDT {}

impl LGDT {
    // TODO: Generic type returns must be investigated. Should I add field for heuristic also ??
    pub fn new() -> Self {
        LGDT {
            tree: None,
            error: None,
        }
    }

    pub fn fit<S, F>(
        structure: &mut S,
        min_sup: Support,
        max_depth: Depth,
        fit_method: F,
    ) -> Tree<NodeData<usize>>
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData<usize>>,
    {
        if max_depth <= 2 {
            fit_method(structure, min_sup, max_depth)
        } else {
            let mut solution_tree: Tree<NodeData<usize>> = Tree::new();

            let root_tree = fit_method(structure, min_sup, max_depth);
            let mut root_attribute = None;

            if let Some(root) = root_tree.get_node(root_tree.get_root_index()) {
                solution_tree.add_root(TreeNode {
                    value: root.value,
                    index: 0,
                    left: 0,
                    right: 0,
                });
                root_attribute = root.value.test;
            }

            let root_index = solution_tree.get_root_index();
            let _ = LGDT::build_tree_recurse(
                structure,
                &mut solution_tree,
                root_index,
                root_attribute,
                min_sup,
                max_depth - 1,
                &fit_method,
            );

            solution_tree
        }
    }

    fn create_child<V>(tree: &mut Tree<NodeData<V>>, parent: TreeIndex, is_left: bool) -> TreeIndex
    where
        V: Bounded + Copy,
    {
        let value: NodeData<V> = NodeData::new();
        let node = TreeNode::new(value);
        tree.add_node(parent, is_left, node)
    }

    fn create_leaf<S>(
        tree: &mut Tree<NodeData<usize>>,
        structure: &mut S,
        parent: TreeIndex,
        is_left: bool,
    ) -> usize
    where
        S: Structure,
    {
        let leaf_index = LGDT::create_child(tree, parent, is_left);
        let classes_support = structure.labels_support();
        let top_class = LGDT::get_top_class(&classes_support);
        let error = LGDT::get_misclassification_error(&classes_support);

        if let Some(leaf) = tree.get_node_mut(leaf_index) {
            leaf.value.metric = error;
            leaf.value.out = Some(top_class)
        }
        error
    }

    fn move_tree<V>(
        dest_tree: &mut Tree<NodeData<V>>,
        dest_index: TreeIndex,
        source_tree: &Tree<NodeData<V>>,
        source_index: TreeIndex,
    ) where
        V: Bounded + Copy,
    {
        if let Some(source_node) = source_tree.get_node(source_index) {
            if let Some(root) = dest_tree.get_node_mut(dest_index) {
                root.value = source_node.value;
            }
            let source_left_index = source_node.left;

            if source_left_index > 0 {
                let mut left_index = 0;
                if let Some(root) = dest_tree.get_node_mut(dest_index) {
                    left_index = root.left;
                    if left_index == 0 {
                        left_index = LGDT::create_child(dest_tree, dest_index, true);
                    }
                }
                LGDT::move_tree(dest_tree, left_index, source_tree, source_left_index)
            }

            let source_right_index = source_node.right;
            if source_right_index > 0 {
                let mut right_index = 0;
                if let Some(root) = dest_tree.get_node_mut(dest_index) {
                    right_index = root.right;
                    if right_index == 0 {
                        right_index = LGDT::create_child(dest_tree, dest_index, false);
                    }
                }
                LGDT::move_tree(dest_tree, right_index, source_tree, source_right_index)
            }
        }
    }

    fn build_tree_recurse<S, F>(
        structure: &mut S,
        tree: &mut Tree<NodeData<usize>>,
        index: TreeIndex,
        next: Option<Attribute>,
        minsup: Support,
        depth: Depth,
        fit_method: &F,
    ) -> usize
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData<usize>>,
    {
        let branches = [false, true];
        return if depth <= 1 {
            let mut parent_error = 0;
            for (i, val) in branches.iter().enumerate() {
                let _ = structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 1);
                let child_error = LGDT::get_tree_metric(&child_tree);

                if child_error == <usize>::MAX {
                    let child_error = LGDT::create_leaf(tree, structure, index, !*val);

                    parent_error += child_error;
                } else {
                    let child_index = LGDT::create_child(tree, index, !*val);
                    LGDT::move_tree(tree, child_index, &child_tree, child_tree.get_root_index());
                    parent_error += child_error;
                }

                structure.backtrack();
            }
            parent_error
        } else {
            let mut parent_error = 0;
            for (i, val) in branches.iter().enumerate() {
                let _ = structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 2);
                let mut child_error = LGDT::get_tree_metric(&child_tree);
                if child_error == <usize>::MAX {
                    child_error = LGDT::create_leaf(tree, structure, index, !*val);
                } else {
                    let child_index = LGDT::create_child(tree, index, !*val);
                    if child_error == 0 {
                        LGDT::move_tree(
                            tree,
                            child_index,
                            &child_tree,
                            child_tree.get_root_index(),
                        );
                    } else if let Some(child) = tree.get_node_mut(child_index) {
                        let mut child_next = None;
                        if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
                            child.value = root.value;
                            child_next = child.value.test;
                        }
                        child_error = LGDT::build_tree_recurse(
                            structure,
                            tree,
                            child_index,
                            child_next,
                            minsup,
                            depth - 1,
                            fit_method,
                        );
                    }
                }
                parent_error += child_error;
                structure.backtrack();
            }
            if let Some(parent) = tree.get_node_mut(index) {
                parent.value.metric = parent_error;
            }
            parent_error
        };
    }
}

#[cfg(test)]
mod lgdt_test {
    use crate::algorithms::algorithm_trait::{Algorithm, Basic};
    use crate::algorithms::lgdt::LGDT;
    use crate::algorithms::murtree::MurTree;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::horizontal_binary_structure::HorizontalBinaryStructure;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;

    #[test]
    fn random() {
        let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let a = LGDT::fit(&mut structure, 1, 4, MurTree::fit);
        let error = LGDT::get_tree_metric(&a);
        a.print();
    }
}
