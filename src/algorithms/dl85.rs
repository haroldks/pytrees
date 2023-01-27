use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::dl85_utils::stop_conditions::StopConditions;
use crate::algorithms::dl85_utils::structs_enums::{Constraints, Statistics};
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::heuristics::Heuristic;
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::caching::trie::{DataTrait, Trie, TrieNode};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, CacheIndex, Depth, Item, Support};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

// TODO : Is not working for generic types because of the trait bounds and the use of external methods

struct DL85<'heur, H, T>
where
    H: Heuristic + ?Sized,
    T: DataTrait + Default,
{
    constraints: Constraints,
    heuristic: &'heur mut H,
    cache: Trie<T>,
    stop_conditions: StopConditions<T>,
    statistics: Statistics,
    run_time: Instant,
}

impl<'heur, H, T> DL85<'heur, H, T>
where
    H: Heuristic + ?Sized,
    T: DataTrait + Default,
{
    pub fn new(
        min_sup: Support,
        max_depth: Depth,
        max_error: usize,
        max_time: usize,
        one_time_sort: bool,
        heuristic: &'heur mut H,
    ) -> Self {
        let constaints = Constraints {
            max_depth,
            min_sup,
            max_error,
            max_time,
            one_time_sort,
        };
        Self {
            constraints: constaints,
            heuristic,
            cache: Trie::new(),
            stop_conditions: StopConditions::default(),
            statistics: Statistics {
                constraints: constaints,
                cache_size: 0,
                tree_error: 0,
                duration: Default::default(),
            },
            run_time: Instant::now(),
        }
    }

    pub fn fit(&mut self, structure: &mut RSparseBitsetStructure) {
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
    }

    pub fn recursion(
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
            // println!("Specialized case");
            // println!("Itemset before: {:?}", itemset);
            let error = self.run_specialized_algorithm(structure, parent_index, itemset);
            // println!("Itemset after: {:?}", itemset);
            return error;
        }

        // Explore the children of the node

        // Get the potential children of the node based on the min_sup
        let mut node_candidates = vec![];
        if parent_attribute == Attribute::MAX {
            node_candidates = candidates.clone()
        } else {
            node_candidates = self.get_node_candidates(structure, parent_attribute, &candidates);
        }

        if node_candidates.is_empty() {
            if let Some(node) = self.cache.get_node_mut(parent_index) {
                node.value.set_as_leaf();
                return node.value.get_node_error();
            }
        }

        // TODO: Allow heuristic to be used for the children of the node for one time
        self.heuristic.compute(structure, &mut node_candidates);

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
            if structure.temp_push((*potential_candidate, 0)) >= self.constraints.min_sup
                && right_support >= self.constraints.min_sup
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
                cache_node.value.set_node_error(source_root.value.error);
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
}

#[cfg(test)]
mod dl85_test {
    use crate::algorithms::dl85::DL85;
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

        let mut algo: DL85<'_, _, Data> =
            DL85::new(1, 5, <usize>::MAX, 5, false, heuristic.as_mut());
        algo.fit(&mut structure);

        if let Some(root) = algo.cache.get_node(algo.cache.get_root_index()) {
            println!("Root: {:?}", root);
        }
        println!("Cache size: {}", algo.cache.len());
    }
}
