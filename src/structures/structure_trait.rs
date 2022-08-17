use crate::dataset::data_trait::Dataset;

trait Structure {
    fn new<T>(data: T) -> Self
    where
        T: Dataset;

    fn num_labels(&self) -> usize;
    fn size(&self) -> usize;
    fn label_instances(label: usize) -> usize;
    fn state(&self); // Return itemset  ref or other and support
    fn support(&self) -> usize;
    fn push();
    fn temp_push();
}
