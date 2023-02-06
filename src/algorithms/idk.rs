use crate::algorithms::algorithm_trait::Basic;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Index, Support};

pub struct IDK {
    tree: Option<Tree<NodeData>>,
    error: Option<usize>,
}

impl Basic for IDK {}

impl Default for IDK {
    fn default() -> Self {
        Self::new()
    }
}

impl IDK {
    // TODO: Generic type returns must be investigated.
    pub fn new() -> Self {
        IDK {
            tree: None,
            error: None,
        }
    }

    pub fn fit<S, F>(structure: &mut S, min_sup: Support, fit_method: F) -> Tree<NodeData>
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData>,
    {
        let mut solution_tree: Tree<NodeData> = Tree::new();

        let root_tree = fit_method(structure, min_sup, 2);
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
        let _ = Self::build_tree_recurse(
            structure,
            &mut solution_tree,
            root_index,
            root_attribute,
            min_sup,
            &fit_method,
        );

        solution_tree
    }

    fn build_tree_recurse<S, F>(
        structure: &mut S,
        tree: &mut Tree<NodeData>,
        index: Index,
        next: Option<Attribute>,
        minsup: Support,
        fit_method: &F,
    ) -> usize
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData>,
    {
        let mut parent_error = 0;
        for (i, val) in [false, true].iter().enumerate() {
            let _ = structure.push((next.unwrap(), i));
            let child_tree = fit_method(structure, minsup, 2);
            let mut child_error = Self::get_tree_error(&child_tree);
            if child_error == <usize>::MAX {
                child_error = Self::create_leaf(tree, structure, index, !*val);
            } else {
                let child_index = Self::create_child(tree, index, !*val);
                if child_error == 0 {
                    Self::move_tree(tree, child_index, &child_tree, child_tree.get_root_index());
                } else if let Some(child) = tree.get_node_mut(child_index) {
                    let mut child_next = None;
                    if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
                        child.value = root.value;
                        child_next = child.value.test;
                    }
                    child_error = Self::build_tree_recurse(
                        structure,
                        tree,
                        child_index,
                        child_next,
                        minsup,
                        fit_method,
                    );
                }
            }
            parent_error += child_error;
            structure.backtrack();
        }
        if let Some(parent) = tree.get_node_mut(index) {
            parent.value.error = parent_error;
        }
        parent_error
    }
}

#[cfg(test)]
mod idk_test {
    use crate::algorithms::algorithm_trait::{Algorithm, Basic};
    use crate::algorithms::idk::IDK;
    use crate::algorithms::info_gain::InfoGain;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;

    #[test]
    fn test_idk() {
        let dataset = BinaryDataset::load("test_data/ionosphere.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);
        let a = IDK::fit(&mut structure, 1, InfoGain::fit);
        let error = IDK::get_tree_error(&a);
    }
}
