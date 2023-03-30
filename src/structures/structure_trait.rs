use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::structures_types::{Bitset, Index, Item, LeafInfo, Position, Support};

pub trait Structure {
    fn num_attributes(&self) -> usize;
    fn num_labels(&self) -> usize;
    fn label_support(&self, label: usize) -> Support;
    fn labels_support(&mut self) -> &[Support];
    fn support(&mut self) -> Support;
    fn get_support(&self) -> Support;
    fn push(&mut self, item: Item) -> Support;
    fn backtrack(&mut self);
    fn temp_push(&mut self, item: Item) -> Support;
    fn reset(&mut self);
    fn get_position(&self) -> &Position;
    fn change_position(&mut self, itemset: &[Item]) -> Support {
        self.reset();
        for item in itemset {
            self.push(*item);
        }
        self.support()
    }
}

pub trait BitsetTrait {
    fn extract_leaf_bitvector(
        &mut self,
        tree: &Tree<NodeData>,
        index: Index,
        position: &mut Vec<Item>,
        collector: &mut Vec<LeafInfo>,
    );

    fn set_state(&mut self, state: &Bitset, position: &Position);
}
