use crate::algorithms::dl85_utils::structs_enums::ReturnCondition;
use crate::structures::caching::trie::{DataTrait, TrieNode};
use crate::structures::structures_types::{Depth, Support};
use std::fmt::Debug;
use std::time::Duration;

#[derive(Default)]
pub struct StopConditions<T>
where
    T: DataTrait + Default + Debug,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StopConditions<T>
where
    T: DataTrait + Default + Debug,
{
    pub(crate) fn check(
        &self,
        node: &mut TrieNode<T>,
        support: Support,
        min_sup: Support,
        current_depth: Depth,
        max_depth: Depth,
        current_time: Duration,
        max_time: usize,
        upper_bound: usize,
    ) -> (bool, ReturnCondition) {
        if self.time_limit_reached(current_time, max_time, upper_bound, node) {
            return (true, ReturnCondition::TimeLimitReached);
        }

        if self.max_depth_reached(current_depth, max_depth, upper_bound, node) {
            return (true, ReturnCondition::MaxDepthReached);
        }

        if self.not_enough_support(support, min_sup, upper_bound, node) {
            return (true, ReturnCondition::NotEnoughSupport);
        }

        if self.lower_bound_constrained(upper_bound, node) {
            return (true, ReturnCondition::LowerBoundConstrained);
        }

        if self.pure_node(upper_bound, node) {
            return (true, ReturnCondition::PureNode);
        }
        (false, ReturnCondition::None)
    }

    pub(crate) fn stop_from_lower_bound(
        &self,
        node: &mut TrieNode<T>,
        actual_upper_bound: usize,
    ) -> (bool, ReturnCondition) {
        if node.value.get_lower_bound() >= actual_upper_bound {
            return (true, ReturnCondition::LowerBoundConstrained);
        }
        if node.value.get_leaf_error() <= node.value.get_lower_bound() {
            node.value.set_as_leaf();
            return (true, ReturnCondition::PureNode);
        }
        (false, ReturnCondition::None)
    }

    fn time_limit_reached(
        &self,
        current_time: Duration,
        max_time: usize,
        actual_upper_bound: usize,
        node: &mut TrieNode<T>,
    ) -> bool {
        match current_time.as_secs() as usize >= max_time {
            true => {
                node.value.set_as_leaf();
                true
            }
            false => false,
        }
    }

    fn lower_bound_constrained(&self, actual_upper_bound: usize, node: &mut TrieNode<T>) -> bool {
        match (node.value.get_lower_bound() >= actual_upper_bound) || (actual_upper_bound == 0) {
            true => true,
            false => false,
        }
    }

    fn max_depth_reached(
        &self,
        depth: Depth,
        max_depth: Depth,
        actual_upper_bound: usize,
        node: &mut TrieNode<T>,
    ) -> bool {
        match depth == max_depth {
            true => {
                node.value.set_as_leaf();
                true
            }
            false => false,
        }
    }

    fn not_enough_support(
        &self,
        support: Support,
        min_sup: Support,
        actual_upper_bound: usize,
        node: &mut TrieNode<T>,
    ) -> bool {
        match support < min_sup * 2 {
            true => {
                node.value.set_as_leaf();
                true
            }
            false => false,
        }
    }

    fn pure_node(&self, actual_upper_bound: usize, node: &mut TrieNode<T>) -> bool {
        match node.value.get_leaf_error() == node.value.get_lower_bound() {
            // TODO : check if this is correct
            true => {
                node.value.set_as_leaf();
                true
            }
            false => false,
        }
    }

    pub fn check_using_discrepancy(
        &self,
        node: &mut TrieNode<T>,
        support: Support,
        min_sup: Support,
        current_depth: Depth,
        max_depth: Depth,
        current_time: Duration,
        max_time: usize,
        upper_bound: usize,
        discrepancy: usize,
        max_discrepancy: usize,
    ) -> (bool, ReturnCondition) {
        if self.time_limit_reached(current_time, max_time, upper_bound, node) {
            node.value.set_discrepancy(discrepancy);
            return (true, ReturnCondition::TimeLimitReached);
        }

        if self.max_depth_reached(current_depth, max_depth, upper_bound, node) {
            node.value.set_discrepancy(discrepancy);
            return (true, ReturnCondition::MaxDepthReached);
        }

        if self.not_enough_support(support, min_sup, upper_bound, node) {
            node.value.set_discrepancy(discrepancy);
            return (true, ReturnCondition::NotEnoughSupport);
        }

        if self.pure_node(upper_bound, node) {
            node.value.set_discrepancy(discrepancy);
            return (true, ReturnCondition::PureNode);
        }
        if node.value.get_discrepancy() >= max_discrepancy
            && upper_bound <= node.value.get_lower_bound()
        {
            node.value.set_as_leaf();
            return (true, ReturnCondition::PureNode); // TODO: Change this to a new enum
        }

        (false, ReturnCondition::None)
    }
}
