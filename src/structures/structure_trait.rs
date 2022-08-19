use crate::dataset::data_trait::Dataset;
use crate::structures::structures_types::{Item, Support};

pub(crate) trait Structure {
    fn num_labels(&self) -> usize;
    fn label_support(&self, label: usize) -> Support;
    fn support(&mut self) -> Support;
    fn push(&mut self, item: Item) -> Support;
    fn backtrack(&mut self);
    fn temp_push(&mut self, item: Item) -> Support;
}
