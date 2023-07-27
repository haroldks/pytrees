// use crate::algorithms::algorithm_trait::Basic;
// use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
// use crate::structures::structure_trait::Structure;
// use crate::structures::structures_types::{Attribute, Depth, Index, Support};
//
// pub struct ParallelLGDTMurTree {
//     tree: Option<Tree<NodeData>>,
//     error: Option<usize>,
// }
//
// // impl Basic for ParallelLGDTMurTree {}
//
// impl Default for ParallelLGDTMurTree {
//     fn default() -> Self {
//         Self::new()
//     }
// }
//
// impl ParallelLGDTMurTree {
//     // TODO: Generic type returns must be investigated.
//     pub fn new() -> Self {
//         ParallelLGDTMurTree {
//             tree: None,
//             error: None,
//         }
//     }
//
//     pub fn fit<S, F>(
//         structure: &mut Vec<S>,
//         min_sup: Support,
//         max_depth: Depth,
//         fit_method: F,
//     ) -> Tree<NodeData>
//     where
//         S: Structure,
//         F: Fn(&mut Vec<S>, Support, Depth) -> Tree<NodeData>,
//     {
//         if max_depth <= 2 {
//             fit_method(structure, min_sup, max_depth)
//         } else {
//             // println!("Max depth : {}", max_depth);
//             let mut solution_tree: Tree<NodeData> = Tree::new();
//
//             let root_tree = fit_method(structure, min_sup, max_depth);
//             let mut root_attribute = None;
//
//             if let Some(root) = root_tree.get_node(root_tree.get_root_index()) {
//                 solution_tree.add_root(TreeNode {
//                     value: root.value,
//                     index: 0,
//                     left: 0,
//                     right: 0,
//                 });
//                 root_attribute = root.value.test;
//             }
//
//             if root_attribute.is_some() {
//                 let root_index = solution_tree.get_root_index();
//                 let _ = ParallelLGDTMurTree::build_tree_recurse(
//                     structure,
//                     &mut solution_tree,
//                     root_index,
//                     root_attribute,
//                     min_sup,
//                     max_depth - 1,
//                     &fit_method,
//                 );
//             }
//
//             solution_tree
//         }
//     }
//
//     fn build_tree_recurse<S, F>(
//         structure: &mut Vec<S>,
//         tree: &mut Tree<NodeData>,
//         index: Index,
//         next: Option<Attribute>,
//         minsup: Support,
//         depth: Depth,
//         fit_method: &F,
//     ) -> usize
//     where
//         S: Structure,
//         F: Fn(&mut Vec<S>, Support, Depth) -> Tree<NodeData>,
//     {
//         return if depth <= 1 {
//             let mut parent_error = 0;
//             for (i, val) in [false, true].iter().enumerate() {
//                 let _ = structure[0].push((next.unwrap(), i));
//                 let child_tree = fit_method(structure, minsup, 1);
//                 let child_error = Self::get_tree_error(&child_tree);
//
//                 if child_error == <usize>::MAX {
//                     let child_error =
//                         ParallelLGDTMurTree::create_leaf(tree, structure, index, !*val);
//
//                     parent_error += child_error;
//                 } else {
//                     let child_index = ParallelLGDTMurTree::create_child(tree, index, !*val);
//                     ParallelLGDTMurTree::move_tree(
//                         tree,
//                         child_index,
//                         &child_tree,
//                         child_tree.get_root_index(),
//                     );
//                     parent_error += child_error;
//                 }
//                 structure[0].backtrack();
//
//                 // for s in structure.iter_mut() {
//                 //     s.backtrack();
//                 // }
//             }
//             if let Some(parent) = tree.get_node_mut(index) {
//                 parent.value.error = parent_error;
//             }
//             parent_error
//         } else {
//             let mut parent_error = 0;
//             for (i, val) in [false, true].iter().enumerate() {
//                 // tree.print();
//                 // if next.is_none(){
//                 // }
//                 // println!("Next: {:?}", (next, i));
//                 // println!("Depth : {}", depth - 1);
//                 for s in structure.iter_mut() {
//                     let _ = s.push((next.unwrap(), i));
//                 }
//
//                 let child_tree = fit_method(structure, minsup, 2);
//                 // child_tree.print();
//                 let mut child_error = ParallelLGDTMurTree::get_tree_error(&child_tree);
//                 if child_error == <usize>::MAX {
//                     child_error = ParallelLGDTMurTree::create_leaf(tree, structure, index, !*val);
//                 } else {
//                     let child_index = ParallelLGDTMurTree::create_child(tree, index, !*val);
//                     if child_error == 0 {
//                         ParallelLGDTMurTree::move_tree(
//                             tree,
//                             child_index,
//                             &child_tree,
//                             child_tree.get_root_index(),
//                         );
//                     } else if let Some(child) = tree.get_node_mut(child_index) {
//                         let mut child_next = None;
//                         if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
//                             child.value = root.value;
//                             child_next = child.value.test;
//                         }
//                         child_error = ParallelLGDTMurTree::build_tree_recurse(
//                             structure,
//                             tree,
//                             child_index,
//                             child_next,
//                             minsup,
//                             depth - 1,
//                             fit_method,
//                         );
//                     }
//                 }
//                 parent_error += child_error;
//                 for s in structure.iter_mut() {
//                     s.backtrack()
//                 }
//             }
//             if let Some(parent) = tree.get_node_mut(index) {
//                 parent.value.error = parent_error;
//             }
//             parent_error
//         };
//     }
//
//     fn create_leaf<S>(
//         tree: &mut Tree<NodeData>,
//         structure: &mut Vec<S>,
//         parent: Index,
//         is_left: bool,
//     ) -> usize
//     where
//         S: Structure,
//     {
//         let leaf_index = Self::create_child(tree, parent, is_left);
//         let classes_support = structure[0].labels_support();
//         let error = Self::get_leaf_error(classes_support);
//         if let Some(leaf) = tree.get_node_mut(leaf_index) {
//             leaf.value.error = error.0;
//             leaf.value.out = Some(error.1)
//         }
//         error.0
//     }
//
//     fn get_leaf_error(classes_support: &[usize]) -> (usize, usize) {
//         let mut max_idx = 0;
//         let mut max_value = 0;
//         let mut total = 0;
//         for (idx, value) in classes_support.iter().enumerate() {
//             total += value;
//             if *value >= max_value {
//                 max_value = *value;
//                 max_idx = idx;
//             }
//         }
//         let error = total - max_value;
//         (error, max_idx)
//     }
//
//     fn get_tree_error(tree: &Tree<NodeData>) -> usize {
//         if let Some(root) = tree.get_node(tree.get_root_index()) {
//             return root.value.error;
//         }
//         <usize>::MAX
//     }
//
//     fn create_child(tree: &mut Tree<NodeData>, parent: Index, is_left: bool) -> Index {
//         let value = NodeData::new();
//         let node = TreeNode::new(value);
//         tree.add_node(parent, is_left, node)
//     }
//
//     fn move_tree(
//         dest_tree: &mut Tree<NodeData>,
//         dest_index: Index,
//         source_tree: &Tree<NodeData>,
//         source_index: Index,
//     ) {
//         if let Some(source_node) = source_tree.get_node(source_index) {
//             if let Some(root) = dest_tree.get_node_mut(dest_index) {
//                 root.value = source_node.value;
//             }
//             let source_left_index = source_node.left;
//
//             if source_left_index > 0 {
//                 let mut left_index = 0;
//                 if let Some(root) = dest_tree.get_node_mut(dest_index) {
//                     left_index = root.left;
//                     if left_index == 0 {
//                         left_index = Self::create_child(dest_tree, dest_index, true);
//                     }
//                 }
//                 Self::move_tree(dest_tree, left_index, source_tree, source_left_index)
//             }
//
//             let source_right_index = source_node.right;
//             if source_right_index > 0 {
//                 let mut right_index = 0;
//                 if let Some(root) = dest_tree.get_node_mut(dest_index) {
//                     right_index = root.right;
//                     if right_index == 0 {
//                         right_index = Self::create_child(dest_tree, dest_index, false);
//                     }
//                 }
//                 Self::move_tree(dest_tree, right_index, source_tree, source_right_index)
//             }
//         }
//     }
// }
//
// #[cfg(test)]
// mod lgdt_test {
//     use crate::algorithms::algorithm_trait::{Algorithm, Basic};
//     use crate::algorithms::info_gain::InfoGain;
//     use crate::algorithms::lgdt::LGDT;
//     use crate::algorithms::murtree::MurTree;
//     use crate::dataset::binary_dataset::BinaryDataset;
//     use crate::dataset::data_trait::Dataset;
//     use crate::structures::bitsets_structure::BitsetStructure;
//     use rand::Rng;
//     #[test]
//     fn test_lgdt_murtree_anneal() {
//         let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
//         let bitset_data = BitsetStructure::format_input_data(&dataset);
//         let mut structure = BitsetStructure::new(&bitset_data);
//
//         let steps = 3;
//         let expected_errors = [151usize, 137, 119, 108, 99, 90, 71, 55, 48, 41];
//
//         for _ in 0..steps {
//             let mut rng = rand::thread_rng();
//             let depth = rng.gen_range(1..11) as usize;
//             let a = LGDT::fit(&mut structure, 1, depth, MurTree::fit);
//             let error = LGDT::get_tree_error(&a);
//             assert_eq!(expected_errors.contains(&error), true);
//         }
//     }
//
//     #[test]
//     fn test_lgdt_info_gain_anneal() {
//         let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
//         let bitset_data = BitsetStructure::format_input_data(&dataset);
//         let mut structure = BitsetStructure::new(&bitset_data);
//
//         let steps = 3;
//         let expected_errors = [152usize, 151, 149, 126, 89, 79, 69, 57, 50];
//         for _ in 0..steps {
//             let mut rng = rand::thread_rng();
//             let depth = rng.gen_range(1..11) as usize;
//
//             let a = LGDT::fit(&mut structure, 1, depth, InfoGain::fit);
//             let error = LGDT::get_tree_error(&a);
//             assert_eq!(expected_errors.contains(&error), true);
//         }
//     }
// }
