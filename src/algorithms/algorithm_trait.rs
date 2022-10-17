use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Item, Support, TreeIndex};
use num_traits::Bounded;

pub(crate) trait Algorithm {
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
        let mut candidates = vec![];
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
    ) -> Vec<Vec<(usize, usize)>>
    where
        S: Structure,
    {
        let size = candidates.len();
        let mut matrix = vec![vec![(0usize, 0usize); size]; size];
        for i in 0..size {
            structure.push((candidates[i], 1));
            let val = structure.labels_support();
            matrix[i][i] = (val[0], val[1]);

            for second in i + 1..size {
                structure.push((candidates[second], 1));
                let val = structure.labels_support();
                structure.backtrack();
                matrix[i][second] = (val[0], val[1]);
                matrix[second][i] = (val[0], val[1]);
            }
            structure.backtrack();
        }
        matrix
    }

    fn get_depth_two_leaves_stats(
        matrix: &Vec<Vec<(usize, usize)>>,
        classes_support: &[usize],
        first: Item,
        second: Item,
    ) -> [usize; 2] {
        match first.1 == 0 {
            true => {
                let mut leaves = [0usize; 2];
                match second.1 == 0 {
                    true => {
                        leaves[0] = classes_support[0] + matrix[first.0][second.0].0
                            - matrix[first.0][first.0].0
                            - matrix[second.0][second.0].0;
                        leaves[1] = classes_support[1] + matrix[first.0][second.0].1
                            - matrix[first.0][first.0].1
                            - matrix[second.0][second.0].1;
                        leaves
                    }
                    false => {
                        leaves[0] = matrix[second.0][second.0].0 - matrix[first.0][second.0].0;
                        leaves[1] = matrix[second.0][second.0].1 - matrix[first.0][second.0].1;
                        leaves
                    }
                }
            }
            false => {
                let mut leaves = [0usize; 2];
                match second.1 == 0 {
                    true => {
                        leaves[0] = matrix[first.0][first.0].0 - matrix[first.0][second.0].0;
                        leaves[1] = matrix[first.0][first.0].1 - matrix[first.0][second.0].1;
                        leaves
                    }
                    false => {
                        leaves[0] = matrix[first.0][second.0].0;
                        leaves[1] = matrix[first.0][second.0].1;
                        leaves
                    }
                }
            }
        }
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

    fn build_tree_recurse(tree: &mut Tree<NodeData>, parent: TreeIndex, depth: Depth) {
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

pub(crate) trait Basic {
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
}
