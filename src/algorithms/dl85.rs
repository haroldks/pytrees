use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::dl85_utils::stop_conditions::StopConditions;
use crate::algorithms::dl85_utils::structs_enums::{Constraints, Specialization, Statistics};
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::heuristics::Heuristic;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::caching::trie::{DataTrait, Trie, TrieNode};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, CacheIndex, Depth, Item, Support};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::time::{Duration, Instant};

// TODO : Is not working for generic types because of the trait bounds and the use of external methods

pub struct DL85<'heur, H, T>
where
    H: Heuristic + ?Sized,
    T: DataTrait + Default + Debug,
{
    constraints: Constraints,
    heuristic: &'heur mut H,
    cache: Trie<T>,
    stop_conditions: StopConditions<T>,
    pub statistics: Statistics,
    tree: Tree<NodeData>,
    run_time: Instant,
}

impl<'heur, H, T> DL85<'heur, H, T>
where
    H: Heuristic + ?Sized,
    T: DataTrait + Default + Debug,
{
    pub fn new(
        min_sup: Support,
        max_depth: Depth,
        max_error: usize,
        max_time: usize,
        specialization: Specialization,
        one_time_sort: bool,
        heuristic: &'heur mut H,
    ) -> Self {
        let constaints = Constraints {
            max_depth,
            min_sup,
            max_error,
            max_time,
            one_time_sort,
            specialization,
        };
        Self {
            constraints: constaints,
            heuristic,
            cache: Trie::new(),
            stop_conditions: StopConditions::default(),
            statistics: Statistics {
                num_attributes: 0,
                num_samples: 0,
                train_distribution: vec![],
                constraints: constaints,
                cache_size: 0,
                tree_error: 0,
                duration: Duration::default(),
            },
            tree: Tree::default(),
            run_time: Instant::now(),
        }
    }

    pub fn fit(&mut self, structure: &mut RSparseBitsetStructure) {
        // Update Statistics structures
        self.statistics.num_attributes = structure.num_attributes();
        self.statistics.train_distribution = structure.labels_support();
        self.statistics.num_samples = structure.support();

        let mut candidates = Vec::new();
        if self.constraints.min_sup == 1 {
            candidates = (0..structure.num_attributes()).collect();
        } else {
            for i in 0..structure.num_attributes() {
                if structure.temp_push((i, 0)) >= self.constraints.min_sup
                    && structure.temp_push((i, 1)) >= self.constraints.min_sup
                {
                    candidates.push(i);
                }
            }
        }

        self.heuristic.compute(structure, &mut candidates); // Will sort the candidates according to the heuristic
        let root_data = T::new();
        let root = TrieNode::new(root_data);
        self.cache.add_root(root);
        let root_index = self.cache.get_root_index();

        let mut itemset = BTreeSet::new();
        self.run_time = Instant::now();
        self.recursion(
            structure,
            0,
            self.constraints.max_error,
            Attribute::MAX,
            &mut itemset,
            candidates,
            root_index,
        );
        self.update_statistics();
        if let Some(root) = self.cache.get_node(self.cache.get_root_index()) {
            if root.value.get_node_error() < <usize>::MAX {
                self.generate_tree();
            }
        }
    }

    fn recursion(
        &mut self,
        structure: &mut RSparseBitsetStructure,
        depth: Depth,
        upper_bound: usize,
        parent_attribute: Attribute,
        itemset: &mut BTreeSet<Item>,
        candidates: Vec<usize>,
        parent_index: CacheIndex,
    ) -> usize {
        // TODO: Check if there is not enough time left (Maybe this can be done outside of the recursion)

        let mut child_upper_bound = upper_bound;
        let current_support = structure.support();
        if let Some(node) = self.cache.get_node_mut(parent_index) {
            if self.stop_conditions.check(
                node,
                current_support,
                self.constraints.min_sup,
                depth,
                self.constraints.max_depth,
                self.run_time.elapsed(),
                self.constraints.max_time,
                child_upper_bound,
            ) {
                return node.value.get_node_error();
            }
        }

        // TODO: Check Lower bound constraints

        // TODO: Depth 2 specialized case

        if self.constraints.max_depth - depth <= 2 {
            match self.constraints.specialization {
                Specialization::Murtree => {
                    return self.run_specialized_algorithm(structure, parent_index, itemset);
                }
                Specialization::None => {}
            }
        }

        // Explore the children of the node

        // Get the potential children of the node based on the min_sup
        let mut node_candidates = vec![];
        node_candidates = self.get_node_candidates(structure, parent_attribute, &candidates);

        if node_candidates.is_empty() {
            if let Some(node) = self.cache.get_node_mut(parent_index) {
                node.value.set_as_leaf();
                return node.value.get_node_error();
            }
        }

        if self.constraints.one_time_sort {
            self.heuristic.compute(structure, &mut node_candidates);
        }

        for child in node_candidates.iter() {
            let mut item = (*child, 0);
            let _ = structure.push(item);
            itemset.insert(item);

            let (_, child_index) = self.find_or_create_in_cache(structure, itemset);
            let left_error = self.recursion(
                structure,
                depth + 1,
                child_upper_bound,
                *child,
                itemset,
                node_candidates.clone(),
                child_index,
            );

            structure.backtrack();
            itemset.remove(&item);

            // If the error is too high, we don't need to explore the right part of the node
            if left_error >= child_upper_bound {
                // println!("should stop here and child was {:?}", (*child, 0));
                continue;
            }

            // Explore right part of the node

            let right_upper_bound = child_upper_bound - left_error;
            let item = (*child, 1);

            let _ = structure.push(item);
            itemset.insert(item);

            let (_, child_index) = self.find_or_create_in_cache(structure, itemset);

            let right_error = self.recursion(
                structure,
                depth + 1,
                right_upper_bound,
                *child,
                itemset,
                node_candidates.clone(),
                child_index,
            );
            structure.backtrack();
            itemset.remove(&item);

            if right_error == <usize>::MAX || left_error == <usize>::MAX {
                continue;
            }

            let feature_error = left_error + right_error;
            if feature_error < child_upper_bound {
                child_upper_bound = feature_error;

                if let Some(parent_node) = self.cache.get_node_mut(parent_index) {
                    parent_node.value.set_node_error(child_upper_bound);
                    parent_node.value.set_test(*child);
                }
            }
        }
        return self
            .cache
            .get_node(parent_index)
            .unwrap()
            .value
            .get_node_error();
    }

    fn get_node_candidates(
        &self,
        structure: &mut RSparseBitsetStructure,
        last_candidate: Attribute,
        candidates: &Vec<Attribute>,
    ) -> Vec<Attribute> {
        let mut node_candidates = Vec::new();
        let support = structure.support();
        for potential_candidate in candidates {
            if *potential_candidate == last_candidate {
                continue;
            }
            let left_support = structure.temp_push((*potential_candidate, 0));
            let right_support = support - left_support;

            if left_support >= self.constraints.min_sup && right_support >= self.constraints.min_sup
            {
                node_candidates.push(*potential_candidate);
            }
        }
        node_candidates
    }

    fn find_or_create_in_cache(
        &mut self,
        structure: &mut RSparseBitsetStructure,
        itemset: &mut BTreeSet<Item>,
    ) -> (bool, CacheIndex) {
        let (is_new, index) = self.cache.find_or_create(itemset.iter());

        if is_new {
            if let Some(node) = self.cache.get_node_mut(index) {
                let classes_support = structure.labels_support();
                let (leaf_error, class) = Self::leaf_error(&classes_support);
                node.value.set_leaf_error(leaf_error);
                node.value.set_class(class)
            }
        }

        (is_new, index)
    }

    fn leaf_error(classes_support: &[usize]) -> (usize, usize) {
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

    fn run_specialized_algorithm(
        &mut self,
        structure: &mut RSparseBitsetStructure,
        index: CacheIndex,
        itemset: &mut BTreeSet<Item>,
    ) -> usize {
        let mut tree = LGDT::fit(structure, self.constraints.min_sup, 2, MurTree::fit);
        let error = LGDT::get_tree_error(&tree);
        // tree.print();
        self.stitch_to_cache(index, &tree, tree.get_root_index(), itemset);
        error
    }

    fn stitch_to_cache(
        &mut self,
        cache_index: CacheIndex,
        tree: &Tree<NodeData>,
        source_index: usize,
        itemset: &mut BTreeSet<Item>,
    ) {
        if let Some(source_root) = tree.get_node(source_index) {
            if let Some(cache_node) = self.cache.get_node_mut(cache_index) {
                // println!("Before {:?}", cache_node);
                // if source_root.value.error == <usize>::MAX {
                //     println!("Something goes wrong");
                //     tree.print();
                // }
                cache_node.value.set_node_error(source_root.value.error);
                cache_node.value.set_leaf_error(source_root.value.error);
                // println!("After {:?}", cache_node);
                // println!("{:?}", source_root);
                if source_root.left == source_root.right {
                    // Case when the rode is a leaf
                    cache_node.value.set_as_leaf();
                    cache_node
                        .value
                        .set_class(source_root.value.out.unwrap_or(<usize>::MAX));
                } else {
                    cache_node
                        .value
                        .set_test(source_root.value.test.unwrap_or(Attribute::MAX));
                }
            }

            let source_left_index = source_root.left;
            if source_left_index > 0 {
                itemset.insert((source_root.value.test.unwrap_or(Attribute::MAX), 0));
                let (_, left_index) = self.cache.find_or_create(itemset.iter());
                self.stitch_to_cache(left_index, tree, source_left_index, itemset);
                itemset.remove(&(source_root.value.test.unwrap_or(Attribute::MAX), 0));
            }

            let source_right_index = source_root.right;
            if source_right_index > 0 {
                itemset.insert((source_root.value.test.unwrap_or(Attribute::MAX), 1));
                let (_, right_index) = self.cache.find_or_create(itemset.iter());
                self.stitch_to_cache(right_index, tree, source_right_index, itemset);
                itemset.remove(&(source_root.value.test.unwrap_or(Attribute::MAX), 1));
            }
        }
    }

    fn update_statistics(&mut self) {
        self.statistics.cache_size = self.cache.len();
        self.statistics.duration = self.run_time.elapsed();
        if let Some(node) = self.cache.get_node(self.cache.get_root_index()) {
            self.statistics.tree_error = node.value.get_node_error();
        }
    }

    fn generate_tree(&mut self) {
        let mut tree = Tree::new();
        let mut path = BTreeSet::new();

        // println!("{:?}", self.cache);

        // Creating root node
        if let Some(cache_node) = self.cache.get_node(self.cache.get_root_index()) {
            let node_data = self.create_node_data(
                cache_node.value.get_test(),
                cache_node.value.get_node_error(),
                cache_node.value.get_class(),
                cache_node.value.is_leaf(),
            );
            let _ = tree.add_root(TreeNode::new(node_data));

            // Creating the rest of the tree
            let root_index = tree.get_root_index();
            self.generate_tree_rec(
                cache_node.value.get_test(),
                &mut path,
                &mut tree,
                root_index,
            );
        }
        self.tree = tree;
    }

    fn generate_tree_rec(
        &self,
        attribute: Attribute,
        path: &mut BTreeSet<Item>,
        tree: &mut Tree<NodeData>,
        parent_index: usize,
    ) {
        if attribute == Attribute::MAX {
            return;
        }

        for i in 0..2 {
            // Creating children
            path.insert((attribute, i));
            // println!("path {:?}", path);
            // println!("path {:?}", self.cache.find(path.iter()));

            if let Some(cache_node_index) = self.cache.find(path.iter()) {
                if let Some(cache_node) = self.cache.get_node(cache_node_index) {
                    // println!("{:?}", cache_node.value.get_test());
                    // println!("{:?}", cache_node.item);

                    // println!("{:?}", cache_node);
                    // println!("{:?}", cache_node.value.get_node_error());
                    // println!("{:?}", cache_node.value.is_leaf());
                    let node_data = self.create_node_data(
                        cache_node.value.get_test(),
                        cache_node.value.get_node_error(),
                        cache_node.value.get_class(),
                        cache_node.value.is_leaf(),
                    );
                    let node_index = tree.add_node(parent_index, i == 0, TreeNode::new(node_data));

                    if !cache_node.value.is_leaf() {
                        self.generate_tree_rec(cache_node.value.get_test(), path, tree, node_index);
                    }
                }
            }
            path.remove(&(attribute, i));
        }
    }

    fn create_node_data(
        &self,
        test: Attribute,
        error: usize,
        out: usize,
        is_leaf: bool,
    ) -> NodeData {
        if is_leaf {
            return NodeData {
                test: None,
                error,
                out: Some(out),
                metric: None,
            };
        }
        NodeData {
            test: Some(test),
            error,
            out: None,
            metric: None,
        }
    }
}

#[cfg(test)]
mod dl85_test {
    use crate::algorithms::dl85::DL85;
    use crate::algorithms::dl85_utils::structs_enums::Specialization;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::heuristics::{Heuristic, InformationGain, NoHeuristic};
    use crate::structures::caching::trie::Data;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
    use crate::structures::structure_trait::Structure;

    #[test]
    fn run_dl85() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);

        let mut heuristic: Box<dyn Heuristic> = Box::new(NoHeuristic::default());

        let mut algo: DL85<'_, _, Data> = DL85::new(
            1,
            6,
            <usize>::MAX,
            10,
            Specialization::None,
            false,
            heuristic.as_mut(),
        );
        algo.fit(&mut structure);
        //
        // if let Some(root) = algo.cache.get_node(algo.cache.get_root_index()) {
        //     println!("Root: {:?}", root);
        // }
        println!("Statistics: {:?}", algo.statistics);
    }
}
