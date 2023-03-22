use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::murtree::MurTree;
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::Index;
use std::slice::Iter;

pub struct CostComplexityPruning {
    tree: Vec<Tree<NodeData>>,
    alphas: Vec<f64>,
}

impl Default for CostComplexityPruning {
    fn default() -> Self {
        Self::new()
    }
}

impl Basic for CostComplexityPruning {}

impl CostComplexityPruning {
    pub fn new() -> Self {
        Self {
            tree: vec![],
            alphas: vec![],
        }
    }

    pub fn prune<S>(tree: &mut Tree<NodeData>, structure: &mut S)
    where
        S: Structure,
    {
        let mut final_trees = vec![];
        let mut final_alphas = vec![];
        while tree.actual_len() > 3 {
            let (diff_trees, sub_trees) = Self::get_diff_subtrees(tree, structure);
            let infos = CostComplexityPruning::get_sub_trees_infos(&sub_trees);
            let alphas = Self::get_alphas(Self::get_tree_error(tree), &infos);
            let min_alpha = *alphas
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            final_trees.push(diff_trees[min_alpha.0].clone());
            final_alphas.push(min_alpha.1);
            *tree = diff_trees[min_alpha.0].clone();
        }
    }

    fn get_diff_subtrees<S>(
        tree: &Tree<NodeData>,
        structure: &mut S,
    ) -> (Vec<Tree<NodeData>>, Vec<Tree<NodeData>>)
    where
        S: Structure,
    {
        let mut trees = vec![];
        let mut subtrees = vec![];
        if let Some(root) = tree.get_node(tree.get_root_index()) {
            Self::tree_conversion_recursion(tree, root.index, structure, &mut trees, &mut subtrees);
        }
        (trees, subtrees)
    }
    fn tree_conversion_recursion<S>(
        tree: &Tree<NodeData>,
        index: Index,
        structure: &mut S,
        trees: &mut Vec<Tree<NodeData>>,
        subtrees: &mut Vec<Tree<NodeData>>,
    ) where
        S: Structure,
    {
        if let Some(node) = tree.get_node(index) {
            if node.left != 0 {
                if let Some(left_node) = tree.get_node(node.left) {
                    if !Self::is_leaf(left_node) {
                        structure.push((node.value.test.unwrap(), 0usize));
                        let mut diff_tree = Self::convert_node_to_leaf(tree, node.left, structure);
                        Self::clean_error(&mut diff_tree);
                        trees.push(diff_tree);

                        let mut sub_tree = MurTree::empty_tree(1);
                        Self::move_tree(&mut sub_tree, 0, tree, node.left);
                        subtrees.push(sub_tree);

                        Self::tree_conversion_recursion(
                            tree, node.left, structure, trees, subtrees,
                        );
                        structure.backtrack();
                    }
                }
            }
            if node.right != 0 {
                if let Some(right_node) = tree.get_node(node.right) {
                    if !Self::is_leaf(right_node) {
                        structure.push((node.value.test.unwrap(), 1usize));
                        let mut diff_tree = Self::convert_node_to_leaf(tree, node.right, structure);
                        Self::clean_error(&mut diff_tree);
                        trees.push(diff_tree);

                        let mut sub_tree = MurTree::empty_tree(1);
                        Self::move_tree(&mut sub_tree, 0, tree, node.right);
                        subtrees.push(sub_tree);

                        Self::tree_conversion_recursion(
                            tree, node.right, structure, trees, subtrees,
                        );
                        structure.backtrack();
                    }
                }
            }
        }
    }

    fn convert_node_to_leaf<S>(
        tree: &Tree<NodeData>,
        node_index: Index,
        structure: &mut S,
    ) -> Tree<NodeData>
    where
        S: Structure,
    {
        let mut node_tree = tree.clone();
        if let Some(node) = node_tree.get_node_mut(node_index) {
            node.left = 0;
            node.right = 0;
            node.value.test = None;
            let classes_supports = structure.labels_support();
            let error = Self::get_leaf_error(classes_supports);
            node.value.error = error.0;
            node.value.out = Some(error.1);
        }
        node_tree
    }

    fn get_sub_trees_infos(trees: &[Tree<NodeData>]) -> Vec<(usize, usize)> {
        trees
            .iter()
            .map(|tree| {
                (
                    Self::get_tree_error(tree),
                    Self::get_leaves_nodes_count(tree),
                )
            })
            .collect::<Vec<(usize, usize)>>()
    }

    fn get_alphas(tree_error: usize, infos: &[(usize, usize)]) -> Vec<(usize, f64)> {
        infos
            .iter()
            .enumerate()
            .map(|(i, (error, size))| (i, (tree_error - error) as f64 / (size - 1) as f64))
            .collect::<Vec<(usize, f64)>>()
    }

    fn get_tree_leaves_indexes(tree: &Tree<NodeData>) -> Vec<Index> {
        let mut leaves = vec![];
        recursion(0, tree, &mut leaves);

        fn recursion(node_index: Index, tree: &Tree<NodeData>, leaves: &mut Vec<Index>) {
            if let Some(node) = tree.get_node(node_index) {
                if node.left == node.right {
                    leaves.push(node_index);
                } else {
                    recursion(node.left, tree, leaves);
                    recursion(node.right, tree, leaves);
                }
            }
        }
        leaves
    }

    fn clean_error(tree: &mut Tree<NodeData>) {
        fn recursion(parent: Index, tree: &mut Tree<NodeData>) -> usize {
            if let Some(node) = tree.get_node(parent) {
                if CostComplexityPruning::is_leaf(node) {
                    return node.value.error;
                }
            }

            let mut left_index = 0;
            let mut right_index = 0;
            let mut parent_error = 0;

            if let Some(node) = tree.get_node(parent) {
                left_index = node.left;
                right_index = node.right;
            }

            if left_index != 0 {
                parent_error += recursion(left_index, tree)
            }
            if right_index != 0 {
                parent_error += recursion(right_index, tree)
            }
            if let Some(node) = tree.get_node_mut(parent) {
                node.value.error = parent_error;
            }
            parent_error
        }

        recursion(0, tree);
    }

    fn get_leaves_nodes_count(tree: &Tree<NodeData>) -> usize {
        Self::get_tree_leaves_indexes(tree).len()
    }

    fn get_internal_nodes(tree: &Tree<NodeData>) -> Vec<Index> {
        let mut nodes = vec![];
        recursion(0, tree, &mut nodes);
        fn recursion(node_index: Index, tree: &Tree<NodeData>, nodes: &mut Vec<Index>) {
            if let Some(node) = tree.get_node(node_index) {
                if node.left != node.right {
                    nodes.push(node_index);
                    if node.left != 0 {
                        recursion(node.left, tree, nodes);
                    }
                    if node.right != 0 {
                        recursion(node.right, tree, nodes);
                    }
                }
            }
        }
        nodes
    }
}

#[cfg(test)]
mod cost_complexity_pruning {
    use crate::algorithms::algorithm_trait::{Algorithm, Basic};
    use crate::post_process::cc_pruning::CostComplexityPruning;

    use crate::algorithms::lgdt::LGDT;
    use crate::algorithms::murtree::MurTree;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::structure_trait::Structure;
    use rand::Rng;

    #[test]
    fn test_leaves_retrieval_depth() {
        let mut rng = rand::thread_rng();
        let depth = rng.gen_range(1..10) as usize;
        let tree = MurTree::empty_tree(depth);
        let leaves_count = CostComplexityPruning::get_leaves_nodes_count(&tree);
        assert_eq!(leaves_count, (2_i32.pow(depth as u32) as usize));
    }

    #[test]
    fn test_internal_nodes_retrieval_depth() {
        let mut rng = rand::thread_rng();
        let depth = rng.gen_range(1..10) as usize;
        let tree = MurTree::empty_tree(depth);
        let nodes = CostComplexityPruning::get_internal_nodes(&tree);
        assert_eq!(nodes.len(), ((2_i32.pow(depth as u32) - 1) as usize));
    }

    #[test]
    fn test_subtrees_conversion() {
        let dataset = BinaryDataset::load("test_data/ionosphere.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let mut tree = LGDT::fit(&mut structure, 1, 10, MurTree::fit);

        structure.reset();
        let (diff_trees, subtrees) =
            CostComplexityPruning::get_diff_subtrees(&tree, &mut structure);

        let infos = CostComplexityPruning::get_sub_trees_infos(&subtrees);
        let error = LGDT::get_tree_error(&tree);
        CostComplexityPruning::prune(&mut tree, &mut structure);
    }
}
