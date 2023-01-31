use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::dl85_utils::slb::{SimilarDatasets, Similarity};
use crate::algorithms::dl85_utils::stop_conditions::StopConditions;
use crate::algorithms::dl85_utils::structs_enums::{
    Constraints, LowerBoundHeuristic, ReturnCondition, Specialization, Statistics,
};
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::heuristics::Heuristic;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::caching::trie::{DataTrait, Trie, TrieNode};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Index, Item, Support};
use std::cmp::{max, min};
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
    pub tree: Tree<NodeData>,
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
        lower_bound: LowerBoundHeuristic,
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
            lower_bound,
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
        let mut root_data = T::new();

        let root_leaf_error = Self::leaf_error(&structure.labels_support());
        root_data.set_node_error(root_leaf_error.0);
        root_data.set_leaf_error(root_leaf_error.0);

        let root = TrieNode::new(root_data);
        self.cache.add_root(root);
        let root_index = self.cache.get_root_index();

        let mut similarity_data = SimilarDatasets::new();

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
            &mut similarity_data,
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
        parent_index: Index,
        similarity_data: &mut SimilarDatasets<T>,
    ) -> (usize, ReturnCondition) {
        // TODO: Check if there is not enough time left (Maybe this can be done outside of the recursion)

        let mut child_upper_bound = upper_bound;
        let current_support = structure.support(); // TODO: Go to get_support ?

        if let Some(node) = self.cache.get_node_mut(parent_index) {
            let return_condition = self.stop_conditions.check(
                node,
                current_support,
                self.constraints.min_sup,
                depth,
                self.constraints.max_depth,
                self.run_time.elapsed(),
                self.constraints.max_time,
                child_upper_bound,
            );

            if return_condition.0 {
                return (node.value.get_node_error(), return_condition.1);
            }
        }

        if let LowerBoundHeuristic::Similarity = self.constraints.lower_bound {
            if let Some(node) = self.cache.get_node_mut(parent_index) {
                let lower_bound = max(
                    node.value.get_lower_bound(),
                    similarity_data.compute_similarity(structure),
                );
                node.value.set_lower_bound(lower_bound);

                let return_condition = self
                    .stop_conditions
                    .stop_from_lower_bound(node, child_upper_bound);
                if return_condition.0 {
                    return (node.value.get_node_error(), return_condition.1);
                }
            }
        }

        if self.constraints.max_depth - depth <= 2 {
            match self.constraints.specialization {
                Specialization::Murtree => {
                    return self.run_specialized_algorithm(structure, parent_index, itemset, depth);
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
                return (node.value.get_node_error(), ReturnCondition::None);
            }
        }

        if !self.constraints.one_time_sort {
            self.heuristic.compute(structure, &mut node_candidates);
        }

        let mut child_similarity_data = SimilarDatasets::new();
        let mut min_lower_bound = <usize>::MAX;

        for child in node_candidates.iter() {
            let mut left_lower_bound = 0;
            let mut right_lower_bound = 0;

            let lower_bounds = self.compute_lower_bounds(
                *child,
                structure,
                itemset,
                &mut child_similarity_data,
                self.constraints.lower_bound,
            );
            left_lower_bound = lower_bounds.0;
            right_lower_bound = lower_bounds.1;

            let mut item = (*child, 0);
            let _ = structure.push(item);
            itemset.insert(item);

            let (_, child_index) = self.find_or_create_in_cache(structure, itemset);

            if let Some(child_node) = self.cache.get_node_mut(child_index) {
                child_node.value.set_lower_bound(left_lower_bound);
            }

            let return_infos = self.recursion(
                structure,
                depth + 1,
                child_upper_bound,
                *child,
                itemset,
                node_candidates.clone(),
                child_index,
                &mut child_similarity_data,
            );

            let left_error = return_infos.0;

            let _ = self.update_similarity_data(
                &mut child_similarity_data,
                structure,
                child_index,
                return_infos.1,
            );

            structure.backtrack();
            itemset.remove(&item);

            // If the error is too high, we don't need to explore the right part of the node
            if left_error as f64 > child_upper_bound as f64 - right_lower_bound as f64 {
                // TODO: Ugly
                if let Some(node) = self.cache.get_node_mut(child_index) {
                    min_lower_bound = match left_error == <usize>::MAX {
                        true => min(
                            min_lower_bound,
                            node.value.get_lower_bound() + right_lower_bound,
                        ),
                        false => min(min_lower_bound, left_error + right_lower_bound),
                    }
                }

                continue;
            }

            // Explore right part of the node

            let right_upper_bound = child_upper_bound - left_error;
            let item = (*child, 1);

            let _ = structure.push(item);
            itemset.insert(item);

            let (_, child_index) = self.find_or_create_in_cache(structure, itemset);

            if let Some(child_node) = self.cache.get_node_mut(child_index) {
                child_node.value.set_lower_bound(right_lower_bound);
            }

            let return_infos = self.recursion(
                structure,
                depth + 1,
                right_upper_bound,
                *child,
                itemset,
                node_candidates.clone(),
                child_index,
                similarity_data,
            );

            let right_error = return_infos.0;

            let _ = self.update_similarity_data(
                &mut child_similarity_data,
                structure,
                child_index,
                return_infos.1,
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
                if child_upper_bound == 0 {
                    break;
                }
            } else {
                min_lower_bound = min(feature_error, min_lower_bound);
            }
        }

        if let Some(node) = self.cache.get_node_mut(parent_index) {
            if node.value.get_node_error() == <usize>::MAX {
                node.value.set_lower_bound(max(
                    node.value.get_lower_bound(),
                    max(min_lower_bound, upper_bound),
                ));
                return (
                    node.value.get_node_error(),
                    ReturnCondition::LowerBoundConstrained,
                );
            }
        }
        return (
            self.cache
                .get_node(parent_index)
                .unwrap()
                .value
                .get_node_error(),
            ReturnCondition::Done,
        );
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
    ) -> (bool, Index) {
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

    fn compute_lower_bounds(
        &self,
        attribute: Attribute,
        structure: &mut RSparseBitsetStructure,
        itemset: &mut BTreeSet<Item>,
        similarities: &mut SimilarDatasets<T>,
        option: LowerBoundHeuristic,
    ) -> (usize, usize) {
        let mut lower_bounds: [usize; 2] = [0, 0];

        for (i, lower_bound) in lower_bounds.iter_mut().enumerate() {
            itemset.insert((attribute, i));
            if let Some(index) = self.cache.find(itemset.iter()) {
                if let Some(node) = self.cache.get_node(index) {
                    *lower_bound = match node.value.get_node_error() == <usize>::MAX {
                        true => node.value.get_lower_bound(),
                        false => node.value.get_node_error(),
                    }
                }
            }
            itemset.remove(&(attribute, i));

            if let LowerBoundHeuristic::Similarity = option {
                structure.push((attribute, i));
                let sim_lb = similarities.compute_similarity(structure);
                *lower_bound = max(*lower_bound, sim_lb);
                structure.backtrack();
            }
        }

        (lower_bounds[0], lower_bounds[1])
    }

    fn update_similarity_data(
        &self,
        similarity_dataset: &mut SimilarDatasets<T>,
        structure: &mut RSparseBitsetStructure,
        child_index: Index,
        condition: ReturnCondition,
    ) -> bool {
        match condition {
            ReturnCondition::LowerBoundConstrained => false,
            _ => {
                if let Some(child_node) = self.cache.get_node(child_index) {
                    similarity_dataset.update(&child_node.value, structure);
                    return true;
                }
                false
            }
        }
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
        index: Index,
        itemset: &mut BTreeSet<Item>,
        depth: Depth,
    ) -> (usize, ReturnCondition) {
        let mut tree = LGDT::fit(structure, self.constraints.min_sup, depth, MurTree::fit);
        let error = LGDT::get_tree_error(&tree);
        self.stitch_to_cache(index, &tree, tree.get_root_index(), itemset);
        (error, ReturnCondition::FromSpecializedAlgorithm)
    }

    fn stitch_to_cache(
        &mut self,
        cache_index: Index,
        tree: &Tree<NodeData>,
        source_index: usize,
        itemset: &mut BTreeSet<Item>,
    ) {
        if let Some(source_root) = tree.get_node(source_index) {
            if let Some(cache_node) = self.cache.get_node_mut(cache_index) {
                cache_node.value.set_node_error(source_root.value.error);
                cache_node.value.set_leaf_error(source_root.value.error);

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

            if let Some(cache_node_index) = self.cache.find(path.iter()) {
                if let Some(cache_node) = self.cache.get_node(cache_node_index) {
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
    use crate::algorithms::dl85_utils::structs_enums::{LowerBoundHeuristic, Specialization};
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
            LowerBoundHeuristic::None,
            false,
            heuristic.as_mut(),
        );
        algo.fit(&mut structure);
    }
}
