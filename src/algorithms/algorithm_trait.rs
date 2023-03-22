use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Index, Item, Support};

pub trait Algorithm {
    fn build_depth_one_tree<S>(structure: &mut S, min_sup: Support) -> Tree<NodeData>
    where
        S: Structure;

    fn build_depth_two_tree<S>(structure: &mut S, min_sup: Support) -> Tree<NodeData>
    where
        S: Structure;

    fn fit<S>(structure: &mut S, min_sup: Support, max_depth: Depth) -> Tree<NodeData>
    where
        S: Structure,
    {
        match max_depth < 2 {
            true => Self::build_depth_one_tree(structure, min_sup),
            false => Self::build_depth_two_tree(structure, min_sup),
        }
    }

    fn generate_candidates_list<S>(structure: &mut S, min_sup: Support) -> Vec<Attribute>
    where
        S: Structure,
    {
        let num_attributes = structure.num_attributes();
        let mut candidates = Vec::with_capacity(num_attributes);
        for i in 0..num_attributes {
            if structure.temp_push((i, 0)) >= min_sup && structure.temp_push((i, 1)) >= min_sup {
                candidates.push(i);
            }
        }
        candidates
    }

    fn build_depth_two_matrix<S>(
        structure: &mut S,
        candidates: &Vec<Attribute>,
    ) -> Vec<Vec<Vec<usize>>>
    where
        S: Structure,
    {
        let size = candidates.len();
        let mut matrix = vec![vec![vec![]; size]; size];
        for i in 0..size {
            structure.push((candidates[i], 1));
            let val = structure.labels_support();
            matrix[i][i] = val.to_vec();

            for second in i + 1..size {
                structure.push((candidates[second], 1));
                let val = structure.labels_support();
                matrix[i][second] = val.to_vec();
                matrix[second][i] = val.to_vec();
                structure.backtrack();
            }
            structure.backtrack();
        }
        matrix
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

    fn empty_tree(depth: Depth) -> Tree<NodeData> {
        let mut tree = Tree::new();
        let value = NodeData::new();
        let node = TreeNode::new(value);
        let root = tree.add_root(node);
        Self::build_tree_recurse(&mut tree, root, depth);
        tree
    }

    fn build_tree_recurse(tree: &mut Tree<NodeData>, parent: Index, depth: Depth) {
        if depth == 0 {
            if let Some(parent_node) = tree.get_node_mut(parent) {
                parent_node.left = 0;
                parent_node.right = 0;
            }
        } else {
            let value = NodeData::new();
            let node = TreeNode::new(value);
            let left = tree.add_node(parent, true, node);
            Self::build_tree_recurse(tree, left, depth - 1);
            let node = TreeNode::new(value);
            let right = tree.add_node(parent, false, node);
            Self::build_tree_recurse(tree, right, depth - 1);
        }
    }

    fn create_leaves(leaf_ref: &mut TreeNode<NodeData>, data: &[usize], error: usize) {
        leaf_ref.value.error = error;
        leaf_ref.value.out = match data[1] > data[0] {
            true => Some(1),
            false => Some(0),
        };
    }
}

pub trait Basic {
    fn is_leaf(node: &TreeNode<NodeData>) -> bool {
        node.left == node.right
    }

    fn get_diff_errors(main_errors: &[usize], sub_errors: &[usize]) -> Vec<usize> {
        let mut errors = vec![];
        for i in main_errors.iter().zip(sub_errors.iter()) {
            errors.push(i.0 - i.1);
        }
        errors
    }

    fn get_leaf_error(classes_support: &[usize]) -> (usize, usize) {
        let mut max_idx = 0;
        let mut max_value = 0;
        let mut total = 0;
        for (idx, value) in classes_support.iter().enumerate() {
            total += value;
            if *value >= max_value {
                max_value = *value;
                max_idx = idx;
            }
        }
        let error = total - max_value;
        (error, max_idx)
    }

    fn get_misclassification_error(classes_support: &[usize]) -> usize {
        classes_support.iter().sum::<usize>() - classes_support.iter().max().unwrap()
    }

    fn get_top_class(classes_support: &[usize]) -> usize {
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

    fn get_tree_error(tree: &Tree<NodeData>) -> usize {
        if let Some(root) = tree.get_node(tree.get_root_index()) {
            return root.value.error;
        }
        <usize>::MAX
    }

    fn create_child(tree: &mut Tree<NodeData>, parent: Index, is_left: bool) -> Index {
        let value = NodeData::new();
        let node = TreeNode::new(value);
        tree.add_node(parent, is_left, node)
    }

    fn create_leaf<S>(
        tree: &mut Tree<NodeData>,
        structure: &mut S,
        parent: Index,
        is_left: bool,
    ) -> usize
    where
        S: Structure,
    {
        let leaf_index = Self::create_child(tree, parent, is_left);
        let classes_support = structure.labels_support();
        let error = Self::get_leaf_error(classes_support);
        if let Some(leaf) = tree.get_node_mut(leaf_index) {
            leaf.value.error = error.0;
            leaf.value.out = Some(error.1)
        }
        error.0
    }

    fn move_tree(
        dest_tree: &mut Tree<NodeData>,
        dest_index: Index,
        source_tree: &Tree<NodeData>,
        source_index: Index,
    ) {
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
                        left_index = Self::create_child(dest_tree, dest_index, true);
                    }
                }
                Self::move_tree(dest_tree, left_index, source_tree, source_left_index)
            }

            let source_right_index = source_node.right;
            if source_right_index > 0 {
                let mut right_index = 0;
                if let Some(root) = dest_tree.get_node_mut(dest_index) {
                    right_index = root.right;
                    if right_index == 0 {
                        right_index = Self::create_child(dest_tree, dest_index, false);
                    }
                }
                Self::move_tree(dest_tree, right_index, source_tree, source_right_index)
            }
        }
    }
}
