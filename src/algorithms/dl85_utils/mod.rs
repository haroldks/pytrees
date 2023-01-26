use crate::structures::caching::trie::{DataTrait, TrieNode};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Depth, Support};

#[derive(Default)]
pub struct StopConditions<T>
where
    T: DataTrait + Default,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StopConditions<T>
where
    T: DataTrait + Default,
{
    pub(crate) fn check(
        &self,
        node: &mut TrieNode<T>,
        support: Support,
        min_sup: Support,
        current_depth: Depth,
        max_depth: Depth,
        upper_bound: usize,
    ) -> bool {
        self.max_depth_reached(current_depth, max_depth, upper_bound, node)
            || self.not_enough_support(support, min_sup, upper_bound, node)
            || self.pure_node(upper_bound, node)
            || self.lower_bound_constrained(upper_bound, node)
    }

    fn lower_bound_constrained(&self, actual_upper_bound: usize, node: &mut TrieNode<T>) -> bool {
        match node.value.get_lower_bound() >= actual_upper_bound {
            true => {
                // println!("Lower bound constrained")
                node.value.set_as_leaf();
                true
            }
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
                // println!("Max depth reached: {}", depth);
                node.value.set_lower_bound(actual_upper_bound);
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
                // println!("Not enough support: {}", support);
                node.value.set_lower_bound(actual_upper_bound);
                node.value.set_as_leaf();
                true
            }
            false => false,
        }
    }

    fn pure_node(&self, actual_upper_bound: usize, node: &mut TrieNode<T>) -> bool {
        match node.value.get_leaf_error() == 0 {
            true => {
                // println!("Pure node");
                node.value.set_lower_bound(actual_upper_bound);
                node.value.set_as_leaf();
                true
            }
            false => false,
        }
    }

    // TODO: Add more stop conditions such as LowerBound
}
