use crate::dataset::data_trait::Dataset;
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{
    Bitset, BitsetStackState, BitsetStructData, Item, Position, Support,
};

#[derive(Clone)]
struct RSparseBitsetStructure<'data> {
    inputs: &'data BitsetStructData,
    support: Support,
    num_labels: usize,
    position: Position,
    state: BitsetStackState,
    index: Vec<usize>,
    limit: Vec<usize>,
}

impl<'data> Structure for RSparseBitsetStructure<'data> {
    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        let support = Support::MAX;
        if label < self.num_labels {
            if let Some(last_state) = self.get_last_state() {
                if let Some(limit) = self.limit.last() {
                    let mut count = 0;
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..*limit + 1 {
                        let cursor = self.index[i];
                        count += (label_bitset[cursor] & last_state[cursor]).count_ones()
                    }
                    return count as Support;
                }
            }
        }
        return support;
    }

    fn support(&mut self) -> Support {
        let mut support = self.support;
        if let Some(last_state) = self.get_last_state() {
            if let Some(limit) = self.limit.last() {
                let mut count = 0;
                for i in 0..*limit + 1 {
                    let cursor = self.index[i];
                    count += last_state[cursor].count_ones();
                }
                support = count as Support;
            }
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
            self.limit.pop();
            self.support();
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        let support = self.push(item);
        self.backtrack();
        support
    }
}

impl<'data> RSparseBitsetStructure<'data> {
    fn format_input_data<T>(data: &T) -> BitsetStructData
    where
        T: Dataset,
    {
        BitsetStructure::format_input_data(data)
    }

    fn new(inputs: &'data BitsetStructData) -> Self {
        let index = (0..inputs.chunks).collect::<Vec<usize>>();

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

        let mut structure = RSparseBitsetStructure {
            inputs,
            support: Support::MAX,
            num_labels: inputs.targets.len(),
            position: vec![],
            state,
            index,
            limit: vec![inputs.chunks - 1],
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
            new_state = last_state.clone();
        }

        if new_state.len() > 0 {
            if let Some(limit) = self.limit.last() {
                let mut limit = *limit;
                let feature_vec = &self.inputs.inputs[item.0];
                for i in (0..limit + 1).rev() {
                    let cursor = self.index[i];
                    let word = match item.1 {
                        0 => new_state[cursor] & !feature_vec[cursor],
                        _ => new_state[cursor] & feature_vec[cursor],
                    };
                    new_state[cursor] = word;
                    if word == 0 {
                        self.index[i] = limit;
                        self.index[limit] = cursor;
                        limit -= 1;
                    }
                }
                self.limit.push(limit);
                self.state.push(new_state);
            }
        }
    }
}

#[cfg(test)]
mod test_rsparse_bitset {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
    use crate::structures::structure_trait::Structure;

    #[test]
    fn load_sparse_bitset() {
        let dataset = BinaryDataset::load("datasets/rsparse_dataset.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);

        assert_eq!(bitset_data.chunks, 3);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 2);
        }
        assert_eq!(
            structure
                .index
                .iter()
                .eq((0..3).collect::<Vec<usize>>().iter()),
            true
        );
    }

    #[test]
    fn compute_stats() {
        let dataset = BinaryDataset::load("datasets/rsparse_dataset.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);

        let expected_support = 192;
        assert_eq!(structure.support, expected_support);

        let expected_label_supports = [64usize, 128];
        assert_eq!(structure.label_support(0), expected_label_supports[0]);
        assert_eq!(structure.label_support(1), expected_label_supports[1]);
    }

    #[test]
    fn branching_in_dataset() {
        let dataset = BinaryDataset::load("datasets/rsparse_dataset.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);

        let support = structure.push((0, 1));

        assert_eq!(support, 128);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 1);
        }

        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 128);
        assert_eq!(structure.label_support(0), 0);

        let support = structure.push((1, 0));

        assert_eq!(support, 64);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 0);
        }
        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 64);
        assert_eq!(structure.label_support(0), 0);

        structure.backtrack();

        assert_eq!(structure.support, 128);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 1);
        }

        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 128);
        assert_eq!(structure.label_support(0), 0);
    }
}
