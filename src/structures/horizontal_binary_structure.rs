use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{
    HBSStackState, HBSState, HorizontalData, Item, Position, Support,
};

#[derive(Clone)]
struct HorizontalBinaryStructure<'data> {
    input: &'data HorizontalData,
    support: Support,
    num_labels: usize,
    position: Position,
    state: HBSStackState,
}

impl<'data> Structure for HorizontalBinaryStructure<'data> {
    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        let mut support = Support::MAX;
        if label < self.num_labels {
            if let Some(state) = self.get_last_state() {
                support = state[label].len();
            }
        }
        support
    }

    fn support(&mut self) -> usize {
        let mut support = self.support;
        if let Some(last) = self.get_last_state() {
            support = last.iter().map(|rows| rows.len()).sum();
        }
        self.support = support;
        support
    }

    fn push(&mut self, item: Item) -> Support {
        self.position.push(item);
        self.pushing(item);
        self.support()
    }

    fn backtrack(&mut self) {
        if self.position.len() > 0 {
            self.position.pop();
            self.state.pop();
            self.support();
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        let support = self.push(item);
        self.backtrack();
        support
    }
}

impl<'data> HorizontalBinaryStructure<'data> {
    fn format_input_data<T>(data: &T) -> HorizontalData
    where
        T: Dataset,
    {
        let data_ref = data.get_train();
        let num_labels = data.num_labels();
        let size = data.train_size();
        let mut inputs = vec![vec![]; num_labels];

        for i in 0..size {
            inputs[data_ref.0[i]].push(data_ref.1[i].clone());
        }

        inputs
    }

    fn new(inputs: &'data HorizontalData) -> Self {
        let mut state = HBSStackState::new();

        let mut initial_state = HBSState::new();
        for i in 0..inputs.len() {
            initial_state.push((0..inputs[i].len()).collect::<Vec<usize>>())
        }
        state.push(initial_state);

        let mut structure = HorizontalBinaryStructure {
            input: inputs,
            support: Support::MAX,
            num_labels: inputs.len(),
            position: vec![],
            state,
        };
        structure.support();
        structure
    }

    fn get_last_state(&self) -> Option<&HBSState> {
        self.state.last()
    }

    fn pushing(&mut self, item: Item) {
        let mut new_state = HBSState::new();
        if let Some(last) = self.get_last_state() {
            for i in 0..self.num_labels {
                let mut label_transactions = vec![];
                for transaction in &last[i] {
                    let input = &self.input[i][*transaction];
                    if input[item.0] == item.1 {
                        label_transactions.push(*transaction);
                    }
                }
                new_state.push(label_transactions);
            }
        }

        self.state.push(new_state);
    }
}

#[cfg(test)]
mod test_horizontal_binary_structure {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::horizontal_binary_structure::HorizontalBinaryStructure;
    use crate::structures::structure_trait::Structure;

    #[test]
    fn load_horizontal_structure() {
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let horizontal_data = HorizontalBinaryStructure::format_input_data(&dataset);
        let data_structure = HorizontalBinaryStructure::new(&horizontal_data);
        let state = [[[0usize, 1], [0, 1]]];
        let input = [[[1usize, 0, 1], [0, 1, 1]], [[0, 0, 0], [0, 1, 0]]];

        assert_eq!(data_structure.position.len(), 0);
        assert_eq!(data_structure.num_labels(), 2);
        assert_eq!(data_structure.state.iter().eq(state.iter()), true);
        assert_eq!(data_structure.input.iter().eq(input.iter()), true);
        assert_eq!(data_structure.label_support(0), 2);
        assert_eq!(data_structure.label_support(1), 2);
    }

    #[test]
    fn moving_one_step() {
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let horizontal_data = HorizontalBinaryStructure::format_input_data(&dataset);
        let mut data_structure = HorizontalBinaryStructure::new(&horizontal_data);
        let position = [(0usize, 0usize)];
        let true_state = vec![vec![1usize], vec![0, 1]];

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
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let horizontal_data = HorizontalBinaryStructure::format_input_data(&dataset);
        let mut data_structure = HorizontalBinaryStructure::new(&horizontal_data);

        let position = [(2usize, 1usize), (0, 1)];
        let real_state = vec![vec![0usize], vec![]];

        data_structure.push((2, 1));
        data_structure.push((0, 1));
        assert_eq!(data_structure.position.len(), 2);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support, 1);
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
        assert_eq!(data_structure.support, 2);
        assert_eq!(data_structure.label_support(0), 2);
        assert_eq!(data_structure.label_support(1), 0);

        let real_state = vec![vec![0usize, 1], vec![]];

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }
    }
}
