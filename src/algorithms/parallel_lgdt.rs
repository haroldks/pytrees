use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::structures;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::{BitsetTrait, Structure};
use crate::structures::structures_types::{
    Attribute, Bitset, Depth, Index, Item, StateCollection, Support,
};
use pyo3::ffi::lenfunc;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::sync::{Arc, Mutex};

pub struct ParallelLGDT {
    tree: Option<Tree<NodeData>>,
    error: Option<usize>,
}

impl Basic for ParallelLGDT {}

impl Default for ParallelLGDT {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelLGDT {
    // TODO: Generic type returns must be investigated.
    pub fn new() -> Self {
        ParallelLGDT {
            tree: None,
            error: None,
        }
    }

    fn remove_below_depth(
        tree: &mut Tree<NodeData>,
        depth: Depth,
        index: Index,
        position: &mut Vec<Item>,
    ) {
        if position.len() == depth {
            if let Some(node) = tree.get_node_mut(index) {
                node.left = 0;
                node.right = 0;
                node.value.test = None;
            }
        } else {
            let mut left_index = 0;
            let mut right_index = 0;
            let mut attribute = None;
            if let Some(node) = tree.get_node_mut(index) {
                left_index = node.left;
                right_index = node.right;
                attribute = node.value.test;
            }
            if left_index != 0 {
                if let Some(attr) = attribute {
                    position.push((attr, 0));
                    Self::remove_below_depth(tree, depth, left_index, position);
                    position.pop();
                }
            }
            if right_index != 0 {
                if let Some(attr) = attribute {
                    position.push((attr, 1));
                    Self::remove_below_depth(tree, depth, right_index, position);
                    position.pop();
                }
            }
        }
    }

    pub fn fit<S, F>(
        structure: &mut S,
        min_sup: Support,
        max_depth: Depth,
        n_threads: usize,
        fit_method: F,
    ) -> Tree<NodeData>
    where
        S: Structure + Clone + Send + structures::structure_trait::BitsetTrait,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData> + Send + Sync,
    {
        let method = Arc::new(fit_method);

        if max_depth <= 3 {
            return LGDT::fit(structure, min_sup, max_depth, method.as_ref());
        }

        let mut tree = LGDT::fit(structure, min_sup, 4, method.as_ref());
        let root_index = tree.get_root_index();
        Self::remove_below_depth(&mut tree, 3, root_index, &mut vec![]);
        let mut state_collection = vec![];
        let mut position = vec![];
        let index = 0;
        structure.extract_leaf_bitvector(&tree, index, &mut position, &mut state_collection);

        let remaining_depth = max_depth - 3;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();
        let output = Arc::new(Mutex::new(tree));
        let main_handler = output.clone();

        let thread_structure = structure.clone();

        pool.scope(move |s| {
            for collection in state_collection {
                if collection.error == 0 {
                    continue;
                }
                let mut sub_struct = thread_structure.clone();
                sub_struct.set_state(&collection.bitset, &collection.position);
                let tree_handler = main_handler.clone();
                let method = method.clone();
                s.spawn(move |_| {
                    let tr = LGDT::fit(&mut sub_struct, min_sup, remaining_depth, method.as_ref());
                    let mut out = tree_handler.lock().unwrap();
                    LGDT::move_tree(&mut out, collection.index, &tr, tr.get_root_index());
                    drop(out);
                });
            }
        });

        let x = output.lock().unwrap();
        x.clone()
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
                let child_error = Self::get_tree_error(&child_tree);

                if child_error == <usize>::MAX {
                    let child_error = Self::create_leaf(tree, structure, index, !*val);

                    parent_error += child_error;
                } else {
                    let child_index = Self::create_child(tree, index, !*val);
                    Self::move_tree(tree, child_index, &child_tree, child_tree.get_root_index());
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
                let _ = structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 2);
                let mut child_error = Self::get_tree_error(&child_tree);
                if child_error == <usize>::MAX {
                    child_error = Self::create_leaf(tree, structure, index, !*val);
                } else {
                    let child_index = Self::create_child(tree, index, !*val);
                    if child_error == 0 {
                        Self::move_tree(
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
                        child_error = Self::build_tree_recurse(
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
mod parallel_lgdt_test {
    use crate::algorithms::algorithm_trait::{Algorithm, Basic};
    use crate::algorithms::dl85_utils::structs_enums::Specialization::Murtree;
    use crate::algorithms::info_gain::InfoGain;
    use crate::algorithms::lgdt::LGDT;
    use crate::algorithms::murtree::MurTree;
    use crate::algorithms::parallel_lgdt::ParallelLGDT;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::structure_trait::{BitsetTrait, Structure};
    use rand::Rng;

    #[test]
    fn test_parrallel_lgdt_murtree_anneal() {
        let dataset = BinaryDataset::load("test_data/german-credit.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let steps = 1;
        let expected_errors = [151usize, 137, 119, 108, 99, 90, 71, 55, 48, 41];

        let a = ParallelLGDT::fit(&mut structure, 5, 7, 8, MurTree::fit);
        println!();
        a.print();
    }
}
