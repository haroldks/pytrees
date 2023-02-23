use crate::algorithms::algorithm_trait::{Algorithm, Basic};
use crate::algorithms::dl85_utils::slb::{SimilarDatasets, Similarity};
use crate::algorithms::dl85_utils::stop_conditions::StopConditions;
use crate::algorithms::dl85_utils::structs_enums::{
    Branching, BranchingType, CacheInit, Constraints, DiscrepancyStrategy, HasIntersected,
    LowerBoundHeuristic, ReturnCondition, Specialization, Statistics,
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

// TODO: Check if it can be consistent with similarity lower bound And Murtree specialization
// TODO: Check if it is possible to use blank implementation to reduce duplicates in code
pub struct LDSDL85<'heur, H, T>
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

impl<'heur, H, T> LDSDL85<'heur, H, T>
where
    H: Heuristic + ?Sized,
    T: DataTrait + Default + Debug,
{
    pub fn new(
        min_sup: Support,
        max_depth: Depth,
        discrepancy_budget: usize,
        discrepancy_strategy: DiscrepancyStrategy,
        max_error: usize,
        max_time: usize,
        specialization: Specialization,
        lower_bound: LowerBoundHeuristic,
        branching: BranchingType,
        cache_init: CacheInit,
        cache_init_size: usize,
        one_time_sort: bool,
        heuristic: &'heur mut H,
    ) -> Self {
        let constraints = Constraints {
            max_depth,
            min_sup,
            max_error,
            max_time,
            one_time_sort,
            specialization,
            lower_bound,
            branching,
            cache_init,
            cache_init_size,
            discrepancy_budget,
            discrepancy_strategy,
        };
        Self {
            constraints,
            heuristic,
            cache: Trie::default(),
            stop_conditions: StopConditions::default(),
            statistics: Statistics {
                num_attributes: 0,
                num_samples: 0,
                train_distribution: [0, 0],
                constraints,
                cache_size: 0,
                tree_error: 0,
                duration: Duration::default(),
            },
            tree: Tree::default(),
            run_time: Instant::now(),
        }
    }

    pub fn fit(&mut self, structure: &mut RSparseBitsetStructure) {
        // BEGIN STEP: Setup everything in the statist Update Statistics structures
        self.statistics.num_attributes = structure.num_attributes();
        let distribution = structure.labels_support();
        self.statistics.train_distribution = [distribution[0], distribution[1]];
        self.statistics.num_samples = structure.support();
        // END STEP : Setup everything in the statistics structures

        // BEGIN STEP: Setup the cache

        self.cache = match self.constraints.cache_init {
            CacheInit::Normal => Trie::default(),
            CacheInit::WithMemoryDynamic => {
                Trie::with_capacity(structure.num_attributes(), self.constraints.max_depth)
            }
            CacheInit::WithMemoryFromUser => {
                Trie::from_user_memory(self.constraints.cache_init_size)
            }
        };
        // END STEP: Setup the cache

        // BEGIN STEP: Load candidates
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
        // END STEP: Load candidates

        // BEGIN STEP: Sort candidates
        self.heuristic.compute(structure, &mut candidates);
        // END STEP: Sort candidates

        // BEGIN STEP: Setup the root
        let mut root_data = T::new();
        let root_leaf_error = Self::leaf_error(structure.labels_support());
        root_data.set_node_error(root_leaf_error.0);
        root_data.set_leaf_error(root_leaf_error.0);
        let root = TrieNode::new(root_data);
        self.cache.add_root(root);
        let root_index = self.cache.get_root_index();
        // END STEP: Setup the root

        let mut similarity_data = SimilarDatasets::new();
        let mut itemset = BTreeSet::new();

        // BEGIN STEP: Compute the discrepancy budget
        let budget = match self.constraints.discrepancy_budget == <usize>::MAX {
            true => {
                Self::compute_discrepancy_limit(candidates.len() - 1, self.constraints.max_depth)
            }
            false => self.constraints.discrepancy_budget,
        };
        self.constraints.discrepancy_budget = budget;
        // END STEP: Compute the discrepancy budget

        // BEGIN STEP: Run the algorithm

        let mut current_budget = 0;
        self.run_time = Instant::now();
        let mut max_error = self.constraints.max_error;
        let mut last_iteration = false;
        while current_budget <= budget {
            self.recursion(
                structure,
                0,
                current_budget,
                max_error,
                (Attribute::MAX, 0),
                &mut itemset,
                &candidates,
                root_index,
                true,
                &mut similarity_data,
            );
            max_error = self.get_tree_error();
            if max_error == 0
                || self.run_time.elapsed().as_secs() as usize >= self.constraints.max_time
            {
                break;
            }

            self.augment_discrepancy(&mut current_budget);

            if current_budget >= budget {
                current_budget = budget;
                last_iteration = true;
            }

            if last_iteration {
                break;
            }

            if current_budget > budget {
                break;
            }
        }
        // END STEP: Run the algorithm

        // BEGIN STEP: Update the statistics
        self.update_statistics();
        if self.get_tree_error() < <usize>::MAX {
            self.generate_tree();
        }
        // END STEP: Update the statistics
    }

    fn compute_discrepancy_limit(nb_candidates: usize, remaining_depth: Depth) -> usize {
        let mut max_discrepancy = nb_candidates;
        for i in 1..remaining_depth {
            max_discrepancy += nb_candidates.saturating_sub(i);
        }
        max_discrepancy
    }

    fn augment_discrepancy(&self, budget: &mut usize) {
        if let DiscrepancyStrategy::Double = self.constraints.discrepancy_strategy {
            if *budget == 0 {
                *budget = 1;
            }
        }

        *budget = match self.constraints.discrepancy_strategy {
            DiscrepancyStrategy::None => 0,
            DiscrepancyStrategy::Incremental => *budget + 1,
            DiscrepancyStrategy::Double => *budget * 2,
        }
    }

    fn recursion(
        &mut self,
        structure: &mut RSparseBitsetStructure,
        depth: Depth,
        current_discrepancy: usize,
        upper_bound: usize,
        parent_item: Item,
        itemset: &mut BTreeSet<Item>,
        candidates: &[usize],
        parent_index: Index,
        parent_is_new: bool,
        similarity_data: &mut SimilarDatasets<T>,
    ) -> (usize, ReturnCondition, HasIntersected) {
        // TODO: Check if there is not enough time left (Maybe this can be done outside of the recursion)

        let mut child_upper_bound = upper_bound;
        let current_support = structure.support();

        // BEGIN STEP: Check if we should stop
        if let Some(node) = self.cache.get_node_mut(parent_index) {
            let return_condition = self.stop_conditions.check_using_discrepancy(
                node,
                current_support,
                self.constraints.min_sup,
                depth,
                self.constraints.max_depth,
                self.run_time.elapsed(),
                self.constraints.max_time,
                child_upper_bound,
                current_discrepancy,
                self.constraints.discrepancy_budget,
            );

            if return_condition.0 {
                return (
                    node.value.get_node_error(),
                    return_condition.1,
                    HasIntersected::No,
                );
            }
        }
        // END STEP: Check if we should stop

        if !parent_is_new {
            let _ = structure.push(parent_item);
        }

        // TODO : Add the option to use it only when similarity branching is used
        // BEGIN STEP: Check if we should use the similarity lower bound to stop
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
                    return (
                        node.value.get_node_error(),
                        return_condition.1,
                        HasIntersected::Yes,
                    );
                }
            }
        }
        // END STEP: Check if we should use the similarity lower bound to stop

        // BEGIN STEP: Check if we should use the murtree specialization and if yes use it
        if self.constraints.max_depth - depth <= 2 {
            match self.constraints.specialization {
                Specialization::Murtree => {
                    return self.run_specialized_algorithm(
                        structure,
                        parent_index,
                        upper_bound,
                        itemset,
                        self.constraints.max_depth - depth,
                    );
                }
                Specialization::None => {}
            }
        }
        // END STEP: Check if we should use the murtree specialization and if yes use it

        // BEGIN STEP: Get the node candidates
        let mut node_candidates = vec![];
        node_candidates = self.get_node_candidates(structure, parent_item.0, candidates);
        // END STEP: Get the node candidates

        // BEGIN STEP: Check if we should stop because there is no more candidates
        if node_candidates.is_empty() {
            if let Some(node) = self.cache.get_node_mut(parent_index) {
                node.value.set_as_leaf();
                return (
                    node.value.get_node_error(),
                    ReturnCondition::None,
                    HasIntersected::Yes,
                );
            }
        }
        // END STEP: Check if we should stop because there is no more candidates

        // BEGIN STEP: Sort the candidates according to the heuristic
        if !self.constraints.one_time_sort {
            self.heuristic.compute(structure, &mut node_candidates);
        }
        // END STEP: Sort the candidates according to the heuristic

        // BEGIN STEP: Update the discrepancy limit
        let max_discrepancy = Self::compute_discrepancy_limit(
            node_candidates.len(),
            self.constraints.max_depth - depth,
        );
        let current_discrepancy = min(current_discrepancy, max_discrepancy);
        // END STEP: Update the discrepancy limit

        // BEGIN STEP: Setup the node similarity data
        let mut child_similarity_data = SimilarDatasets::new();
        let mut min_lower_bound = <usize>::MAX;
        // END STEP: Setup the node similarity data

        let mut discrepancy_pruned = false;

        // BEGIN STEP: Explore the candidates
        for (position, child) in node_candidates.iter().enumerate() {
            // BEGIN STEP: Check if we should stop because we have reached the discrepancy budget
            if position > current_discrepancy {
                discrepancy_pruned = true;
                break;
            }
            // END STEP: Check if we should stop because we have reached the discrepancy budget

            // BEGIN STEP: Choose where to branch first
            let branching_data = self.find_where_to_branch_first(
                *child,
                itemset,
                structure,
                &mut child_similarity_data,
            );

            let first = &branching_data[0];
            let second = &branching_data[1];
            // END STEP: Choose where to branch first

            // BEGIN STEP: Setup the first child node
            let mut item = (*child, first.branch);
            itemset.insert(item);

            // TODO: Check if this is the best place to do this

            let (is_new, child_index) = self.cache.find_or_create(itemset.iter());
            if is_new {
                let _ = structure.push(item);
                self.init_data(structure, child_index);
            }

            if let Some(child_node) = self.cache.get_node_mut(child_index) {
                child_node.value.set_lower_bound(first.lower_bound);
            }
            // END STEP: Setup the first child node

            // BEGIN STEP: Explore the first child node
            let return_infos = self.recursion(
                structure,
                depth + 1,
                current_discrepancy - position,
                child_upper_bound,
                item,
                itemset,
                &node_candidates,
                child_index,
                is_new,
                &mut child_similarity_data,
            );
            let left_error = return_infos.0;
            // END STEP: Explore the first child node

            // BEGIN STEP: Will backtrack if the node is new and update the similarity data if needed

            self.reset_after_branching(
                structure,
                itemset,
                is_new,
                &item,
                &return_infos,
                child_index,
                similarity_data,
            );

            // END STEP: will backtrack if the node is new and update the similarity data if needed

            // BEGIN STEP: If the error is too high, we don't need to explore the right part of the node
            if left_error as f64 >= child_upper_bound as f64 - second.lower_bound as f64 {
                // TODO: Ugly
                if let Some(node) = self.cache.get_node_mut(child_index) {
                    min_lower_bound = match left_error == <usize>::MAX {
                        true => min(
                            min_lower_bound,
                            node.value.get_lower_bound() + second.lower_bound,
                        ),
                        false => min(min_lower_bound, left_error + second.lower_bound),
                    }
                }

                continue;
            }
            // END STEP: If the error is too high, we don't need to explore the right part of the node

            // BEGIN STEP: Setup the second child node
            let right_upper_bound = child_upper_bound - left_error;
            let item = (*child, second.branch);
            itemset.insert(item);

            let (is_new, child_index) = self.cache.find_or_create(itemset.iter());
            if is_new {
                let _ = structure.push(item);
                self.init_data(structure, child_index);
            }

            if let Some(child_node) = self.cache.get_node_mut(child_index) {
                child_node.value.set_lower_bound(second.lower_bound);
            }
            // END STEP: Setup the second child node

            // BEGIN STEP: Explore the second child node
            let return_infos = self.recursion(
                structure,
                depth + 1,
                current_discrepancy - position,
                right_upper_bound,
                item,
                itemset,
                &node_candidates,
                child_index,
                is_new,
                similarity_data,
            );
            let right_error = return_infos.0;
            // END STEP: Explore the second child node

            // BEGIN STEP: Will backtrack if the node is new and update the similarity data if needed
            self.reset_after_branching(
                structure,
                itemset,
                is_new,
                &item,
                &return_infos,
                child_index,
                similarity_data,
            );
            // END STEP: Will backtrack if the node is new and update the similarity data if needed

            if right_error == <usize>::MAX || left_error == <usize>::MAX {
                continue;
            }

            // BEGIN STEP: Update the node error if possible based on the upper bound and the branches error
            let feature_error = left_error + right_error;
            if feature_error < child_upper_bound {
                child_upper_bound = feature_error;

                if let Some(parent_node) = self.cache.get_node_mut(parent_index) {
                    parent_node.value.set_node_error(child_upper_bound);

                    parent_node.value.set_test(*child);

                    if parent_node.value.get_lower_bound() == child_upper_bound {
                        break;
                    }
                }
            } else {
                min_lower_bound = min(feature_error, min_lower_bound);
            }
            // END STEP: Update the node error if possible based on the upper bound and the branches error
        }

        // BEGIN STEP: If the node error is still MAX, we need to update the lower bound
        if let Some(node) = self.cache.get_node_mut(parent_index) {
            if node.value.get_node_error() == <usize>::MAX {
                node.value.set_lower_bound(max(
                    node.value.get_lower_bound(),
                    max(min_lower_bound, upper_bound),
                ));
                return (
                    node.value.get_node_error(),
                    ReturnCondition::LowerBoundConstrained,
                    HasIntersected::Yes,
                );
            }
            if node.value.get_node_error() == 0 || !discrepancy_pruned {
                node.value
                    .set_discrepancy(self.constraints.discrepancy_budget);
            }
        }
        return (
            self.cache
                .get_node(parent_index)
                .unwrap()
                .value
                .get_node_error(),
            ReturnCondition::Done,
            HasIntersected::Yes,
        );
        // END STEP: If the node error is still MAX, we need to update the lower bound
    }

    fn get_node_candidates(
        &self,
        structure: &mut RSparseBitsetStructure,
        last_candidate: Attribute,
        candidates: &[Attribute],
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

    fn init_data(&mut self, structure: &mut RSparseBitsetStructure, index: Index) {
        if let Some(node) = self.cache.get_node_mut(index) {
            let classes_support = structure.labels_support();
            let (leaf_error, class) = Self::leaf_error(classes_support);
            node.value.set_leaf_error(leaf_error);
            node.value.set_class(class)
        }
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

    fn find_where_to_branch_first(
        &self,
        child: Attribute,
        itemset: &mut BTreeSet<Item>,
        structure: &mut RSparseBitsetStructure,
        similarity_dataset: &mut SimilarDatasets<T>,
    ) -> [Branching; 2] {
        let mut lower_bounds = [0, 0];
        // If Dynamic branching is enabled, we check where to move first
        if let BranchingType::Dynamic = self.constraints.branching {
            for (i, lower_bound) in lower_bounds.iter_mut().enumerate() {
                itemset.insert((child, i));
                if let Some(index) = self.cache.find(itemset.iter()) {
                    if let Some(node) = self.cache.get_node(index) {
                        let error = node.value.get_node_error();
                        *lower_bound = match error < <usize>::MAX {
                            true => error,
                            false => node.value.get_lower_bound(),
                        }
                    }
                }
                itemset.remove(&(child, i));
            }

            if let LowerBoundHeuristic::Similarity = self.constraints.lower_bound {
                let similarity_lower_bounds = self.compute_lower_bounds(
                    child,
                    structure,
                    itemset,
                    similarity_dataset,
                    self.constraints.lower_bound,
                );

                lower_bounds[0] = max(lower_bounds[0], similarity_lower_bounds.0);
                lower_bounds[1] = max(lower_bounds[1], similarity_lower_bounds.1);
            }
        }

        let first_item = match lower_bounds[1] > lower_bounds[0] {
            true => 1usize,
            false => 0,
        };

        let second_item = (first_item + 1) % 2;

        let first_lower_bound = match first_item == 0 {
            true => lower_bounds[0],
            false => lower_bounds[1],
        };

        let second_lower_bound = match second_item == 0 {
            true => lower_bounds[0],
            false => lower_bounds[1],
        };

        let first_data = Branching {
            branch: first_item,
            lower_bound: first_lower_bound,
        };
        let second_data = Branching {
            branch: second_item,
            lower_bound: second_lower_bound,
        };
        [first_data, second_data]
    }

    fn reset_after_branching(
        &self,
        structure: &mut RSparseBitsetStructure,
        itemset: &mut BTreeSet<Item>,
        is_new: bool,
        item: &Item,
        return_infos: &(usize, ReturnCondition, HasIntersected),
        child_index: usize,
        child_similarity_data: &mut SimilarDatasets<T>,
    ) {
        let has_intersected = match return_infos.2 {
            HasIntersected::Yes => true,
            HasIntersected::No => false,
        };
        itemset.remove(item);
        if is_new || has_intersected {
            if let LowerBoundHeuristic::Similarity = self.constraints.lower_bound {
                let _ = self.update_similarity_data(
                    child_similarity_data,
                    structure,
                    child_index,
                    return_infos.1,
                );
            }
            structure.backtrack();
        } else if let LowerBoundHeuristic::Similarity = self.constraints.lower_bound {
            structure.push(*item);
            let _ = self.update_similarity_data(
                child_similarity_data,
                structure,
                child_index,
                return_infos.1,
            );
            structure.backtrack();
        }
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
        upper_bound: usize,
        itemset: &mut BTreeSet<Item>,
        depth: Depth,
    ) -> (usize, ReturnCondition, HasIntersected) {
        if let Some(node) = self.cache.get_node(index) {
            if upper_bound < node.value.get_lower_bound() {
                return (
                    node.value.get_node_error(),
                    ReturnCondition::LowerBoundConstrained,
                    HasIntersected::Yes,
                );
            }
        }

        let mut tree = LGDT::fit(structure, self.constraints.min_sup, depth, MurTree::fit);
        let error = LGDT::get_tree_error(&tree);
        self.stitch_to_cache(index, &tree, tree.get_root_index(), itemset);
        (
            error,
            ReturnCondition::FromSpecializedAlgorithm,
            HasIntersected::Yes,
        )
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

    fn get_tree_error(&self) -> usize {
        if let Some(root) = self.cache.get_node(self.cache.get_root_index()) {
            root.value.get_node_error()
        } else {
            <usize>::MAX
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
    use crate::algorithms::dl85_utils::structs_enums::{
        BranchingType, CacheInit, LowerBoundHeuristic, Specialization,
    };
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
            BranchingType::Dynamic,
            CacheInit::Normal,
            0,
            false,
            heuristic.as_mut(),
        );
        algo.fit(&mut structure);
    }
}
