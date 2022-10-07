use crate::structures::structures_types::{Item, Support};

pub trait Structure {
    // TODO: Add capacity control on the structures to avoid memory relocation
    fn num_attributes(&self) -> usize;
    fn num_labels(&self) -> usize;
    fn label_support(&self, label: usize) -> Support;
    fn labels_support(&self) -> Vec<Support>;
    fn support(&mut self) -> Support;
    fn push(&mut self, item: Item) -> Support;
    fn backtrack(&mut self);
    fn temp_push(&mut self, item: Item) -> Support;
    fn reset(&mut self);
    fn change_position(&mut self, itemset: &Vec<Item>) -> Support {
        self.reset();
        for item in itemset {
            self.push(*item);
        }
        self.support()
    }
}
