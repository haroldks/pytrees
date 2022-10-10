use crate::algorithms::algorithm_trait::Algorithm;
use crate::algorithms::murtree::MurTree;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Support, TreeIndex};

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
        // TODO use new methods
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

    fn move_tree(
        dest_tree: &mut Tree<NodeData<usize>>,
        dest_index: TreeIndex,
        source_tree: &Tree<NodeData<usize>>,
        source_index: TreeIndex,
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
    ) where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData<usize>>,
    {
        let branches = [false, true];
        if depth <= 1 {
            for (i, val) in branches.iter().enumerate() {
                let support = structure.push((next.unwrap(), i));
                if support > 2 * minsup {
                    // TODO just convert to leaves here should fix the main issue
                    let child_index = LGDT::create_child(tree, index, !*val); // TODO: Move it after child tree generation
                    let child_tree = fit_method(structure, minsup, 1);
                    println!("{:?}", structure.get_position());
                    println!();
                    println!();
                    if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
                        if root.value.metric < <usize>::MAX {
                            LGDT::move_tree(
                                tree,
                                child_index,
                                &child_tree,
                                child_tree.get_root_index(),
                            );
                        }
                        if root.value.metric == <usize>::MAX {
                            let classes_support = structure.labels_support();
                            let out = MurTree::get_top_class(&classes_support); // TODO move them out of murtree
                            let error = MurTree::get_misclassification_error(&classes_support);
                            let mut data: NodeData<usize> = NodeData::new();
                            data.out = Some(out);
                            data.metric = error;
                            let node: TreeNode<NodeData<usize>> = TreeNode::new(data);
                            let mut node_tree = Tree::new();
                            node_tree.add_root(node);
                            LGDT::move_tree(
                                tree,
                                child_index,
                                &node_tree,
                                node_tree.get_root_index(),
                            );
                        }
                    }
                }

                structure.backtrack();
            }
        } else {
            let mut error = 0;
            for (i, val) in branches.iter().enumerate() {
                let support = structure.push((next.unwrap(), i));
                // println!("Position {:?}", structure.get_position());
                // println!("Support {:?}", structure.support());

                if support > 2 * minsup {
                    // TODO: Shady condition to remove or to ascertain
                    // TODO just convert to leaves here should fix the main issue

                    let child_index = LGDT::create_child(tree, index, !*val); // TODO: Move it after child tree generation
                    println!("Position {:?}", structure.get_position());
                    println!("Support {:?}", structure.support());
                    let child_tree = fit_method(structure, minsup, 2);
                    let mut child_next = None;

                    child_tree.print();
                    println!();
                    println!();
                    println!();
                    println!();
                    tree.print();
                    let mut child_error = 0usize;
                    if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
                        child_error = root.value.metric;

                        if child_error == 0 {
                            LGDT::move_tree(
                                tree,
                                child_index,
                                &child_tree,
                                child_tree.get_root_index(),
                            );
                        } else if let Some(child) = tree.get_node_mut(child_index) {
                            child.value = root.value;
                            child_next = child.value.test;
                        }
                    }
                    if child_error > 0 && child_error < <usize>::MAX {
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
                    if child_error == <usize>::MAX {
                        let classes_support = structure.labels_support();
                        let out = MurTree::get_top_class(&classes_support);
                        let error = MurTree::get_misclassification_error(&classes_support);
                        let mut data: NodeData<usize> = NodeData::new();
                        data.out = Some(out);
                        data.metric = error;
                        let node: TreeNode<NodeData<usize>> = TreeNode::new(data);
                        let mut node_tree = Tree::new();
                        node_tree.add_root(node);
                        LGDT::move_tree(tree, child_index, &node_tree, node_tree.get_root_index());
                    }
                    if (0..<usize>::MAX).contains(&child_error) {
                        error += child_error;
                    }
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

        let a = LGDT::fit(&mut structure, 1, 10, MurTree::fit); // TODO : Fix case when leaf error is already 0;
                                                                // TODO: I can do it by getting the leaf error from the child if it is 0 I stop the search for this part directly
        a.print();
    }
}
