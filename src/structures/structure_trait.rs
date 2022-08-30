use crate::structures::structures_types::{Item, Support};

pub trait Structure {
    fn num_attributes(&self) -> usize;
    fn num_labels(&self) -> usize;
    fn label_support(&self, label: usize) -> Support;
    fn labels_support(&self) -> Vec<Support>;
    fn support(&mut self) -> Support;
    fn push(&mut self, item: Item) -> Support;
    fn backtrack(&mut self);
    fn temp_push(&mut self, item: Item) -> Support;
}
