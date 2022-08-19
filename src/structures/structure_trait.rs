use crate::dataset::data_trait::Dataset;
use crate::structures::structures_types::{HBSState, Item, Support};

pub(crate) trait Structure {
    fn new<T>(data: T) -> Self
    where
        T: Dataset;

    fn num_labels(&self) -> usize;
    fn label_support(&self, label: usize) -> Support;
    fn get_last_state(&self) -> Option<&HBSState>; // Return itemset  ref or other and support
    fn support(&mut self) -> Support;
    fn push(&mut self, item: Item) -> Support;
    fn backtrack(&mut self);
    fn temp_push(&mut self, item: Item) -> Support;
}
