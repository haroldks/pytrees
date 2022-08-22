use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{
    Bitset, BitsetStackState, BitsetStructData, Item, Position, Support,
};

#[derive(Clone)]
struct BitsetStructure<'data> {
    inputs: &'data BitsetStructData,
    support: Support,
    num_labels: usize,
    position: Position,
    state: BitsetStackState,
}

impl<'data> Structure for BitsetStructure<'data> {
    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        let support = Support::MAX;
        if label < self.num_labels {
            if let Some(state) = self.get_last_state() {
                let mut count = 0;
                let label_bitset = &self.inputs.targets[label];
                for (i, label_chunk) in label_bitset.iter().enumerate() {
                    count += (*label_chunk & state[i]).count_ones();
                    return count as Support;
                }
            }
        }
        support
    }

    fn support(&mut self) -> Support {
        let mut support = self.support;
        if let Some(current_state) = self.get_last_state() {
            support = current_state
                .iter()
                .map(|long| long.count_ones())
                .sum::<u32>() as Support;
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
        if !self.position.is_empty() {
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

impl<'data> BitsetStructure<'data> {
    fn format_input_data<T>(data: &T) -> BitsetStructData
    where
        T: Dataset,
    {
        let data_ref = data.get_train();
        let num_labels = data.num_labels();
        let size = data.train_size();
        let num_attributes = data.num_attributes();

        let mut chunks = 1usize;
        if size > 64 {
            chunks = match size % 64 {
                0 => size / 64,
                _ => (size / 64) + 1,
            };
        }

        let mut inputs = vec![vec![0u64; chunks]; num_attributes];
        let mut targets = vec![vec![0u64; chunks]; num_labels];

        for (tid, row) in data_ref.1.iter().rev().enumerate() {
            let row_chunk = chunks - 1 - tid / 64;
            for (i, val) in row.iter().enumerate() {
                if *val == 1 {
                    inputs[i][row_chunk] |= 1u64 << (tid % 64);
                }
            }
            let class = data_ref.0[size - 1 - tid];
            targets[class][row_chunk] |= 1u64 << (tid % 64);
        }

        BitsetStructData {
            inputs,
            targets,
            chunks,
            size,
        }
    }

    fn new(inputs: &'data BitsetStructData) -> Self {
        let mut state = BitsetStackState::new();
        let mut inital_state: Bitset = vec![<u64>::MAX; inputs.chunks];

        if !(inputs.size % 64 == 0) {
            let first_dead_bit = 64 - (inputs.chunks * 64 - inputs.size);
            let first_chunk = &mut inital_state[0];

            for i in (first_dead_bit..64).rev() {
                let int_mask = 1u64 << i;
                *first_chunk &= !int_mask;
            }
        }

        state.push(inital_state);

        let mut structure = BitsetStructure {
            inputs,
            support: Support::MAX,
            num_labels: inputs.targets.len(),
            position: vec![],
            state,
        };

        structure.support();
        structure
    }

    fn get_last_state(&self) -> Option<&Bitset> {
        self.state.last()
    }

    fn pushing(&mut self, item: Item) {
        let mut new_state = Bitset::new();
        if let Some(last_state) = self.get_last_state() {
            let feature_vec = &self.inputs.inputs[item.0];
            for (i, long) in feature_vec.iter().enumerate() {
                new_state.push(match item.1 {
                    0 => last_state[i] & !*long,
                    _ => last_state[i] & *long,
                })
            }
        }
        self.state.push(new_state)
    }
}

#[cfg(test)]
mod test_bitsets {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::structure_trait::Structure;

    #[test]
    fn build_bitset_data() {
        let dataset = BinaryDataset::load("datasets/small_.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
    }

    #[test]
    fn load_bitset_data_on_simple_small() {
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);

        let expected_inputs = [[8u64], [5], [12]];
        assert_eq!(bitset_data.inputs.iter().eq(expected_inputs.iter()), true);

        let expected_targets = [[12u64], [3]];
        assert_eq!(bitset_data.targets.iter().eq(expected_targets.iter()), true);

        assert_eq!(bitset_data.chunks, 1);
    }

    #[test]
    fn load_bitset_data_on_another_small() {
        let dataset = BinaryDataset::load("datasets/small_.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);

        let expected_inputs = [[654u64], [214], [108], [197]];
        assert_eq!(bitset_data.inputs.iter().eq(expected_inputs.iter()), true);

        let expected_targets = [[230u64], [793]];
        assert_eq!(bitset_data.targets.iter().eq(expected_targets.iter()), true);

        assert_eq!(bitset_data.chunks, 1);
    }

    #[test]
    fn create_data_structure() {
        let dataset = BinaryDataset::load("datasets/small_.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let structure = BitsetStructure::new(&bitset_data);

        assert_eq!(structure.support, 10);
        assert_eq!(structure.position.is_empty(), true);

        if let Some(state) = structure.get_last_state() {
            let expected_state = [1023u64];
            assert_eq!((state.iter().eq(expected_state.iter())), true);
        }
    }

    #[test]
    fn check_backtracking() {
        let dataset = BinaryDataset::load("datasets/small_.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        structure.push((3, 1));
        structure.push((0, 0));
        structure.backtrack();

        let expected_position = [(3usize, 1usize)];

        assert_eq!(structure.position.iter().eq(expected_position.iter()), true);
        assert_eq!(structure.support, 4);
        if let Some(state) = structure.get_last_state() {
            let expected_state = [197u64];
            assert_eq!((state.iter().eq(expected_state.iter())), true);
        }
    }

    #[test]
    fn moving_on_step() {
        let dataset = BinaryDataset::load("datasets/small_.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let num_attributes = bitset_data.inputs.len();
        let expected_supports = [5usize, 5, 6, 6];

        for i in 0..num_attributes {
            let support = structure.push((i, 0));
            structure.backtrack();
            assert_eq!(support, expected_supports[i]);
        }
    }

    #[test]
    fn check_label_support() {
        let dataset = BinaryDataset::load("datasets/small_.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let itemset = [(0usize, 1usize), (2, 0), (3, 0)];
        for item in &itemset {
            structure.push(*item);
        }

        assert_eq!(structure.support, 2);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 1);
    }
}
