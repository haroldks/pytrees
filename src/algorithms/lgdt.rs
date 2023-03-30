use crate::algorithms::algorithm_trait::Basic;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Index, Support};

pub struct LGDT {
    tree: Option<Tree<NodeData>>,
    error: Option<usize>,
}

impl Basic for LGDT {}

impl Default for LGDT {
    fn default() -> Self {
        Self::new()
    }
}

impl LGDT {
    // TODO: Generic type returns must be investigated.
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
    ) -> Tree<NodeData>
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData>,
    {
        if max_depth <= 2 {
            fit_method(structure, min_sup, max_depth)
        } else {
            // println!("Max depth : {}", max_depth);
            let mut solution_tree: Tree<NodeData> = Tree::new();

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

            if root_attribute.is_some() {
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
            }

            solution_tree
        }
    }

    fn build_tree_recurse<S, F>(
        structure: &mut S,
        tree: &mut Tree<NodeData>,
        index: Index,
        next: Option<Attribute>,
        minsup: Support,
        depth: Depth,
        fit_method: &F,
    ) -> usize
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData>,
    {
        return if depth <= 1 {
            let mut parent_error = 0;
            for (i, val) in [false, true].iter().enumerate() {
                let _ = structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 1);
                let child_error = LGDT::get_tree_error(&child_tree);

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
            if let Some(parent) = tree.get_node_mut(index) {
                parent.value.error = parent_error;
            }
            parent_error
        } else {
            let mut parent_error = 0;
            for (i, val) in [false, true].iter().enumerate() {
                // tree.print();
                // if next.is_none(){
                // }
                // println!("Next: {:?}", (next, i));
                // println!("Depth : {}", depth - 1);
                let x = structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 2);
                // child_tree.print();
                let mut child_error = LGDT::get_tree_error(&child_tree);
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
                parent.value.error = parent_error;
            }
            parent_error
        };
    }
}

#[cfg(test)]
mod lgdt_test {
    use crate::algorithms::algorithm_trait::{Algorithm, Basic};
    use crate::algorithms::info_gain::InfoGain;
    use crate::algorithms::lgdt::LGDT;
    use crate::algorithms::murtree::MurTree;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use rand::Rng;
    #[test]
    fn test_lgdt_murtree_anneal() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let steps = 3;
        let expected_errors = [151usize, 137, 119, 108, 99, 90, 71, 55, 48, 41];

        for _ in 0..steps {
            let mut rng = rand::thread_rng();
            let depth = rng.gen_range(1..11) as usize;
            let a = LGDT::fit(&mut structure, 1, depth, MurTree::fit);
            let error = LGDT::get_tree_error(&a);
            assert_eq!(expected_errors.contains(&error), true);
        }
    }

    #[test]
    fn test_lgdt_info_gain_anneal() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let steps = 3;
        let expected_errors = [152usize, 151, 149, 126, 89, 79, 69, 57, 50];
        for _ in 0..steps {
            let mut rng = rand::thread_rng();
            let depth = rng.gen_range(1..11) as usize;

            let a = LGDT::fit(&mut structure, 1, depth, InfoGain::fit);
            let error = LGDT::get_tree_error(&a);
            assert_eq!(expected_errors.contains(&error), true);
        }
    }
}
