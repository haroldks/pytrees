use crate::algorithms::algorithm_trait::Algorithm;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Item, Support, TreeIndex};

struct LGDT {
    tree: Option<Tree<NodeData<usize>>>,
    error: Option<usize>,
}

impl LGDT {
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
            return fit_method(structure, min_sup, max_depth);
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
            // println!("{:?}", root_attribute);
            LGDT::build_tree_recurse(
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

    fn create_child(
        tree: &mut Tree<NodeData<usize>>,
        parent: TreeIndex,
        is_left: bool,
    ) -> TreeIndex {
        tree.add_node(
            parent,
            is_left,
            TreeNode {
                value: NodeData {
                    test: None,
                    metric: <usize>::MAX,
                    out: None,
                },
                index: 0,
                left: 0,
                right: 0,
            },
        )
    }

    fn move_tree(tree: &mut Tree<NodeData<usize>>, to: TreeIndex, source: &Tree<NodeData<usize>>) {
        if let Some(source_node) = source.get_node(source.get_root_index()) {
            if let Some(root) = tree.get_node_mut(to) {
                root.value = source_node.value;
            }
            let source_left_index = source_node.left;
            let left_index = LGDT::create_child(tree, to, true);
            if let Some(left) = tree.get_node_mut(left_index) {
                if let Some(source_left_node) = source.get_node(source_left_index) {
                    left.value = source_left_node.value;
                }
            }
            let source_right_index = source_node.right;
            let right_index = LGDT::create_child(tree, to, false);
            if let Some(right) = tree.get_node_mut(right_index) {
                if let Some(source_right_node) = source.get_node(source_right_index) {
                    right.value = source_right_node.value;
                }
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
    ) where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData<usize>>,
    {
        let branches = [false, true];
        if depth <= 1 {
            for (i, val) in branches.iter().enumerate() {
                let child_index = LGDT::create_child(tree, index, !*val);
                structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 1);
                LGDT::move_tree(tree, child_index, &child_tree);
                structure.backtrack();
            }
        } else {
            let mut error = 0;
            for (i, val) in branches.iter().enumerate() {
                let child_index = LGDT::create_child(tree, index, !*val);
                structure.push((next.unwrap(), i));
                let child_tree = fit_method(structure, minsup, 2);
                let mut child_next = None;
                // child_tree.print();
                let mut child_error = 0usize;
                if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
                    child_error = root.value.metric;
                    println!("Here {}", root.value.metric);
                    if child_error == 0 {
                        println!();
                        child_tree.print();

                        // tree.print();
                        // println!("Here");
                        println!();
                        println!();

                        tree.print();
                        LGDT::move_tree(tree, child_index, &child_tree);
                        println!();
                        println!();
                        tree.print();
                    } else {
                        if let Some(child) = tree.get_node_mut(child_index) {
                            child.value = root.value;
                            child_next = child.value.test;
                        }
                    }
                }
                if child_error > 0 {
                    LGDT::build_tree_recurse(
                        structure,
                        tree,
                        child_index,
                        child_next,
                        minsup,
                        depth - 1,
                        fit_method,
                    );
                }
                structure.backtrack();
            }
            if let Some(parent) = tree.get_node_mut(index) {
                parent.value.metric = error;
            }
        }
    }
}

#[cfg(test)]
mod lgdt_test {
    use crate::algorithms::algorithm_trait::Algorithm;
    use crate::algorithms::lgdt::LGDT;
    use crate::algorithms::murtree::MurTree;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::horizontal_binary_structure::HorizontalBinaryStructure;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;

    #[test]
    fn random() {
        // TODO: fix when there is only one chunk issue
        let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let a = LGDT::fit(&mut structure, 1, 6, MurTree::fit); // TODO : Fix case when leaf error is already 0;
                                                               // TODO: I can do it by getting the leaf error from the child if it is 0 I stop the search for this part directly
        a.print();
    }
}
