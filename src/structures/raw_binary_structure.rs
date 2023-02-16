use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Item, Position, Support};
use ndarray::s;

#[derive(Clone)]
pub struct RawBinaryStructure<'data> {
    input: &'data BinaryDataset,
    support: Support,
    labels_support: Vec<Support>,
    num_labels: usize,
    num_attributes: usize,
    position: Position,
    state: Vec<Vec<usize>>,
}

impl<'data> Structure for RawBinaryStructure<'data> {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        let mut support = Support::MAX;
        if label < self.num_labels {
            let train = self.input.get_train();
            if let Some(state) = self.get_last_state() {
                support = state.iter().filter(|x| train.0[**x] == label).count();
            }
        }
        support
    }

    fn labels_support(&mut self) -> &[Support] {
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }

        for label in 0..self.num_labels {
            self.labels_support.push(self.label_support(label))
        }
        &self.labels_support
    }

    fn support(&mut self) -> Support {
        if self.support < Support::MAX {
            return self.support;
        }
        if let Some(last) = self.get_last_state() {
            self.support = last.len();
        }

        self.support
    }

    fn get_support(&self) -> Support {
        self.support
    }

    fn push(&mut self, item: Item) -> Support {
        self.position.push(item);
        self.pushing(item);
        self.support()
    }

    fn backtrack(&mut self) {
        if !self.position.is_empty() {
            self.position.pop();
            self.state.pop();
            self.support = Support::MAX;
            self.labels_support.clear();
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        let support = self.push(item);
        self.backtrack();
        support
    }

    fn reset(&mut self) {
        let mut state = Vec::with_capacity(self.num_attributes);
        state.push(self.state[0].clone());
        self.state = state;
        self.position = Vec::with_capacity(self.num_attributes);
        self.support = self.input.size();
        self.labels_support.clear();
    }

    fn get_position(&self) -> &Position {
        &self.position
    }
}

impl<'data> RawBinaryStructure<'data> {
    pub fn new(inputs: &'data BinaryDataset) -> Self {
        let train_size = inputs.train_size();
        let initial_state = (0..train_size).collect::<Vec<usize>>();
        let mut state = Vec::with_capacity(inputs.num_attributes());
        state.push(initial_state);
        Self {
            input: inputs,
            support: Support::MAX,
            labels_support: Vec::with_capacity(inputs.num_labels()),
            num_labels: inputs.num_labels(),
            num_attributes: inputs.num_attributes(),
            position: Vec::with_capacity(inputs.num_attributes()),
            state,
        }
    }

    fn get_last_state(&self) -> Option<&Vec<usize>> {
        self.state.last()
    }

    fn pushing(&mut self, item: Item) {
        let mut new_state = vec![];
        self.support = 0;
        self.labels_support.clear();
        for label in 0..self.num_labels {
            self.labels_support.push(0);
        }
        if let Some(last) = self.state.last() {
            let inputs = &self.input.get_train().1;
            let target = &self.input.get_train().0;
            for tid in last {
                if inputs[*tid][item.0] == item.1 {
                    new_state.push(*tid);
                    self.support += 1;
                    self.labels_support[target[*tid]] += 1;
                }
            }
        }
        self.state.push(new_state);
    }
}

#[cfg(test)]
mod test_raw_binary_structure {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::raw_binary_structure::RawBinaryStructure;
    use crate::structures::structure_trait::Structure;

    #[test]
    fn test() {
        let dataset = BinaryDataset::load("test_data/small_.txt", false, 0.0);
        let mut raw_structure = RawBinaryStructure::new(&dataset);
        let support = raw_structure.support();
    }

    #[test]
    fn moving_one_step() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let mut data_structure = RawBinaryStructure::new(&dataset);

        let position = [(0usize, 0usize)];
        let true_state = vec![1usize, 2, 3];

        data_structure.push((0, 0));
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support, 3);
        assert_eq!(data_structure.label_support(0), 1);
        assert_eq!(data_structure.label_support(1), 2);

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(true_state.iter()), true);
        }
    }
    #[test]
    fn backtracking() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let mut data_structure = RawBinaryStructure::new(&dataset);

        let position = [(2usize, 1usize), (0, 1)];
        let real_state = vec![0];

        data_structure.push((2, 1));
        data_structure.push((0, 1));
        assert_eq!(data_structure.position.len(), 2);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support(), 1);
        assert_eq!(data_structure.label_support(0), 1);
        assert_eq!(data_structure.label_support(1), 0);
        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }

        data_structure.backtrack();
        let position = [position[0]];
        assert_eq!(data_structure.position.len(), 1);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support(), 2);
        assert_eq!(data_structure.label_support(0), 2);
        assert_eq!(data_structure.label_support(1), 0);

        let real_state = vec![0, 1];

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }
    }
}
