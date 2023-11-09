use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, DoublePointerData, Item, Position, Support};
use search_trail::{
    BoolManager, ReversibleBool, ReversibleUsize, SaveAndRestore, StateManager, UsizeManager,
};

const NONE: usize = usize::MAX;
pub struct Part {
    // TODO: Add an Iterator for this part
    // pub(crate) elements : &'elem Vec<usize>,
    pub(crate) begin: usize,
    pub(crate) end: usize,
}

struct State(usize, usize, usize, bool, usize);

struct DoublePointerStructure<'data> {
    input: &'data DoublePointerData,
    tids: Vec<usize>,
    support: Support,
    num_labels: usize,
    num_attributes: usize,
    labels_support: Vec<Support>,
    position: Position,
    state: [ReversibleUsize; 3],
    is_left: ReversibleBool,
    distance: ReversibleUsize, // Steps to restore to attain the initial state
    manager: StateManager,
}

impl<'data> Structure for DoublePointerStructure<'data> {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        let is_left = self.manager.get_bool(self.is_left);
        let (start, end) = self.get_borders();
        let mut support = 0;
        for tid in self.tids[start..end].iter() {
            if self.input.target[*tid] == label {
                support += 1;
            }
        }
        support
    }

    fn labels_support(&mut self) -> &[Support] {
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }

        self.labels_support.clear();
        for label in 0..self.num_labels {
            self.labels_support.push(0);
        }

        // Getting the concerned border for the current state
        let (start, end) = self.get_borders();

        // Looping over elements between the borders in tid vector
        for tid in self.tids[start..end].iter() {
            let label = self.input.target[*tid];
            self.labels_support[label] += 1;
        }
        &self.labels_support
    }

    fn support(&mut self) -> Support {
        if !self.support == Support::MAX {
            return self.support;
        }
        let is_left = self.manager.get_bool(self.is_left);
        let (start, end) = self.get_borders();
        if start >= end {
            self.support = 0;
            return 0;
        }
        let support = end - start;
        self.support
    }

    fn get_support(&self) -> Support {
        self.support
    }

    fn push(&mut self, item: Item) -> Support {
        self.position.push(item);
        let current_distance = self.manager.get_usize(self.distance);
        self.manager.set_usize(self.distance, current_distance + 1);
        self.manager.save_state();
        let statue_value = self.pushing(item);
        self.push_state(statue_value);
        self.support
    }

    fn backtrack(&mut self) {
        if !self.position.is_empty() {
            self.position.pop();
            self.manager.restore_state();
            self.labels_support.clear();
            self.support = <Support>::MAX;
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        let statue_value = self.pushing(item);
        statue_value.4
    }

    fn reset(&mut self) {
        self.position.clear();
        let distance = self.manager.get_usize(self.distance);
        for _ in 0..distance + 1 {
            self.manager.restore_state();
        }
        self.support = <Support>::MAX;
        self.labels_support.clear();
    }

    fn get_position(&self) -> &Position {
        &self.position
    }

    fn get_tids(&self) -> Vec<usize> {
        if self.position.is_empty() {
            return self.tids.clone();
        }
        let (start, end) = self.get_borders();
        self.tids[start..end].to_vec()
    }
}

impl<'data> DoublePointerStructure<'data> {
    pub fn format_input_data<T>(data: &T) -> DoublePointerData
    // TODO: Cancel cloning
    where
        T: Dataset,
    {
        let data_ref = data.get_train();

        let target = data_ref.0.clone();
        let num_labels = data.num_labels();
        let num_attributes = data.num_attributes();
        let mut inputs = vec![Vec::with_capacity(data.train_size()); num_attributes];
        for row in data_ref.1.iter() {
            for (i, val) in row.iter().enumerate() {
                inputs[i].push(*val);
            }
        }

        DoublePointerData {
            inputs,
            target,
            num_labels,
            num_attributes,
        }
    }

    pub fn new(input: &'data DoublePointerData) -> Self {
        let support = input.target.len();
        let tids = (0..support).collect::<Vec<usize>>();
        let mut manager = StateManager::default();
        let mut state = [
            manager.manage_usize(0),
            manager.manage_usize(tids.len()),
            manager.manage_usize(tids.len()),
        ];
        let is_left = manager.manage_bool(true);
        let distance = manager.manage_usize(0);

        manager.save_state(); // Save the initial state

        Self {
            input,
            tids,
            support,
            num_labels: input.num_labels,
            num_attributes: input.num_attributes,
            labels_support: Vec::with_capacity(input.num_labels),
            position: vec![],
            state,
            is_left,
            distance,
            manager,
        }
    }

    fn pushing(&mut self, item: Item) -> State {
        self.labels_support.clear();
        let (start, end) = self.get_borders();
        let is_left = item.1 == 0;
        let mut i = start;
        let mut j = end;
        loop {
            while i < j && self.input.inputs[item.0][self.tids[i]] == 0 {
                i += 1
            }

            if i + 1 >= j {
                break;
            }

            while j > i {
                j -= 1; // Decrements j before checking
                if self.input.inputs[item.0][self.tids[j]] != 1 {
                    break;
                }
            }

            if i == j {
                break;
            }

            self.tids.swap(i, j);
            i += 1;
        }

        let mut support = 0;
        if is_left {
            support = i - start;
        } else {
            support = end - i;
        }
        State(start, i, end, is_left, support)
    }

    fn push_state(&mut self, state_value: State) {
        self.manager.set_usize(self.state[0], state_value.0);
        self.manager.set_usize(self.state[1], state_value.1);
        self.manager.set_usize(self.state[2], state_value.2);
        self.manager.set_bool(self.is_left, state_value.3);
        self.support = state_value.4;
    }

    fn get_borders(&self) -> (usize, usize) {
        let is_left = self.manager.get_bool(self.is_left);
        match is_left {
            true => (
                self.manager.get_usize(self.state[0]),
                self.manager.get_usize(self.state[1]),
            ),
            false => (
                self.manager.get_usize(self.state[1]),
                self.manager.get_usize(self.state[2]),
            ), // FIXME: Check if this is correct
        }
    }
    fn get_state(&self) -> [usize; 3] {
        let state = &self.state;
        let mut state_value = [0; 3];
        for (i, elem) in state.iter().enumerate() {
            state_value[i] = self.manager.get_usize(*elem);
        }
        state_value
    }
}

#[cfg(test)]
mod test_double_pointer {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::double_pointer::DoublePointerStructure;
    use crate::structures::structure_trait::Structure;
    use search_trail::{BoolManager, UsizeManager};

    #[test]
    fn load_double_pointer() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let data = [[1usize, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0]];
        let target = [0usize, 0, 1, 1];
        assert_eq!(bitset_data.inputs.iter().eq(data.iter()), true);
        assert_eq!(bitset_data.target.iter().eq(target.iter()), true);
        assert_eq!(bitset_data.num_labels, 2);
        assert_eq!(bitset_data.num_attributes, 3);
    }

    #[test]
    fn checking_inside_values() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);
        assert_eq!(structure.num_labels(), 2);
        assert_eq!(structure.num_attributes(), 3);
        assert_eq!(structure.tids, [0, 1, 2, 3]);

        // Getting manager
        let manager = &structure.manager;
        let state = &structure.state;
        let is_left = &structure.is_left;
        let pos_value = manager.get_bool(*is_left);
        assert_eq!(pos_value, true);
        let start = manager.get_usize(state[0]);
        let middle = manager.get_usize(state[1]);
        let end = manager.get_usize(state[2]);
        assert_eq!(start, 0);
        assert_eq!(middle, 4);
        assert_eq!(end, 4);
        assert_eq!(structure.position.is_empty(), true);
    }
    #[test]
    fn checking_root_data_structure() {
        let dataset = BinaryDataset::load("test_data/small_.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);
        assert_eq!(structure.support(), 10);
        assert_eq!(structure.label_support(0), 5);
        assert_eq!(structure.label_support(1), 5);
        assert_eq!(structure.labels_support(), &[5, 5]);
    }

    #[test]
    fn moving_from_root_once() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);

        let item = (0, 0);
        structure.push(item);

        assert_eq!(structure.tids, [3, 1, 2, 0]);
        let manager = &structure.manager;
        let state = &structure.state;
        let mut state_value = vec![];
        let is_left = manager.get_bool(structure.is_left);
        assert_eq!(is_left, true);

        for elem in state.iter() {
            state_value.push(manager.get_usize(*elem));
        }
        assert_eq!(state_value, [0, 3, 4]);

        assert_eq!(structure.support(), 3);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 2);
        assert_eq!(structure.labels_support(), &[1, 2]);
    }

    #[test]
    fn moving_from_root_more_than_once() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);

        let item = (0, 0);
        structure.push(item);

        let item = (1, 0);
        structure.push(item);

        assert_eq!(structure.tids, [2, 1, 3, 0]);

        let mut support = structure.support();
        assert_eq!(support, 1);
        assert_eq!(structure.label_support(0), 0);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support(), &[0, 1]);
        let state = structure.get_state();

        structure.push((2, 0));
        assert_eq!(structure.tids, [2, 1, 3, 0]);

        support = structure.support();
        let state = structure.get_state();

        assert_eq!(support, 1);
        assert_eq!(structure.label_support(0), 0);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support(), &[0, 1]);
    }

    #[test]
    fn moving_towards_null_support() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);

        let item = (0, 0);
        structure.push(item);

        let item = (1, 1);
        let support = structure.push(item);

        assert_eq!(support, 2);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support(), &[1, 1]);

        let item = (2, 0);
        let support = structure.push(item);

        assert_eq!(support, 1);
    }

    #[test]
    fn testing_temp_push() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);
        let num_attributes = structure.num_attributes();

        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
        assert_eq!(structure.temp_push((43, 1)), 26);
        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
        assert_eq!(structure.temp_push((43, 0)), 786);
        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
    }

    #[test]
    fn test_backtracking() {
        let dataset = BinaryDataset::load("test_data/small_.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);

        structure.push((3, 1));
        let s = structure.temp_push((2, 1));
        assert_eq!(s, 2);

        structure.push((0, 0));
        structure.backtrack();

        let expected_position = [(3usize, 1usize)];

        assert_eq!(structure.position.iter().eq(expected_position.iter()), true);
        assert_eq!(structure.support, <usize>::MAX);
    }
    #[test]
    fn moving_on_step_and_backtrack() {
        let dataset = BinaryDataset::load("test_data/small_.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);

        let num_attributes = bitset_data.inputs.len();
        let expected_supports = [5usize, 5, 6, 6];

        for i in 0..num_attributes {
            let support = structure.push((i, 0));
            structure.backtrack();
            assert_eq!(support, expected_supports[i]);
        }
    }

    #[test]
    fn see_tids() {
        let dataset = BinaryDataset::load("test_data/rsparse_dataset.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);
        // Print in binary

        let sup = structure.push((0, 0));
        println!("Support: {}", sup);

        println!("Tids: {:?}", structure.get_tids());
    }
}
