use crate::dataset::data_trait::Dataset;
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{
    Bitset, BitsetStackState, BitsetStructData, Item, Position, Support,
};

#[derive(Clone)]
pub struct RSparseBitsetStructure<'data> {
    inputs: &'data BitsetStructData,
    support: Support,
    num_labels: usize,
    num_attributes: usize,
    position: Position,
    state: BitsetStackState,
    index: Vec<usize>,
    limit: Vec<isize>,
}

impl<'data> Structure for RSparseBitsetStructure<'data> {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        let state = &self.state;
        let support = Support::MAX;

        if label < self.num_labels {
            if let Some(limit) = self.limit.last() {
                let mut count = 0;
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        if let Some(val) = state[cursor].last() {
                            count += (label_bitset[cursor] & val).count_ones()
                        }
                    }
                }
                return count as Support;
            }
        }
        return support;
    }

    fn labels_support(&self) -> Vec<Support> {
        let state = &self.state;
        let mut support = Vec::with_capacity(self.num_labels);
        if let Some(limit) = self.limit.last() {
            for label in 0..self.num_labels {
                let mut count = 0;
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        if let Some(val) = state[cursor].last() {
                            count += (label_bitset[cursor] & val).count_ones()
                        }
                    }
                }
                support.push(count as Support);
            }
        }
        support
    }

    fn support(&mut self) -> Support {
        let state = &self.state;
        let mut support = self.support;
        if let Some(limit) = self.limit.last() {
            let mut count = 0;
            if *limit >= 0 {
                for i in 0..(*limit + 1) as usize {
                    let cursor = self.index[i];
                    if let Some(val) = state[cursor].last() {
                        count += val.count_ones();
                    }
                }
            }
            support = count as Support;
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
            if self.is_empty() {
                self.limit.pop();
            } else {
                if let Some(limit) = self.limit.last() {
                    for i in 0..(*limit + 1) as usize {
                        self.state[self.index[i]].pop();
                    }
                    self.limit.pop();
                }
            }

            self.support();
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        let support = self.push(item);
        self.backtrack();
        support
    }

    fn reset(&mut self) {
        self.position = Vec::with_capacity(self.num_attributes);
        self.limit = Vec::with_capacity(self.num_attributes);
        self.limit.push((self.inputs.chunks - 1) as isize);
        let state = self
            .state
            .iter()
            .map(|stack| vec![stack[0]])
            .collect::<Vec<Bitset>>();
        self.state = state;
        self.support();
    }
    fn get_position(&self) -> &Position {
        &self.position
    }
}

impl<'data> RSparseBitsetStructure<'data> {
    pub fn format_input_data<T>(data: &T) -> BitsetStructData
    where
        T: Dataset,
    {
        BitsetStructure::format_input_data(data)
    }

    pub fn new(inputs: &'data BitsetStructData) -> Self {
        let index = (0..inputs.chunks).collect::<Vec<usize>>();

        let num_attributes = inputs.inputs.len();
        let mut state: Vec<Bitset> = vec![Vec::with_capacity(num_attributes); inputs.chunks];
        for i in 0..inputs.chunks {
            state[i].push(u64::MAX);
        }

        if inputs.size % 64 != 0 {
            let first_dead_bit = 64 - (inputs.chunks * 64 - inputs.size);
            let first_chunk = &mut state[0];

            for i in (first_dead_bit..64).rev() {
                let int_mask = 1u64 << i;
                first_chunk[0] &= !int_mask;
            }
        }

        let mut limit = Vec::with_capacity(num_attributes);
        limit.push((inputs.chunks - 1) as isize);

        let mut structure = RSparseBitsetStructure {
            inputs,
            support: Support::MAX,
            num_labels: inputs.targets.len(),
            num_attributes,
            position: Vec::with_capacity(num_attributes),
            state,
            index,
            limit,
        };

        structure.support();
        structure
    }

    fn get_state(&self) -> &BitsetStackState {
        &self.state
    }

    fn is_empty(&self) -> bool {
        if let Some(limit) = self.limit.last() {
            return *limit < 0;
        }
        false
    }

    fn pushing(&mut self, item: Item) {
        if let Some(limit) = self.limit.last() {
            let mut limit = *limit;
            if limit >= 0 {
                let feature_vec = &self.inputs.inputs[item.0];
                let mut lim = limit as usize;
                for i in (0..lim + 1).rev() {
                    let cursor = self.index[i];
                    if let Some(val) = self.state[cursor].last() {
                        let word = match item.1 {
                            0 => val & !feature_vec[cursor],
                            _ => val & feature_vec[cursor],
                        };
                        if word == 0 {
                            self.index[i] = self.index[lim];
                            self.index[lim] = cursor;
                            limit -= 1;
                            lim = lim.saturating_sub(1);
                            if limit < 0 {
                                break;
                            }
                        } else {
                            if self.position.len() == 2 {}
                            self.state[cursor].push(word);
                        }
                    }
                }
            }

            self.limit.push(limit)
        }
    }
}

#[cfg(test)]
mod test_rsparse_bitset {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
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

    #[test]
    fn compute_state_on_small_dataset() {
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);
        let num_attributes = structure.num_attributes();

        let support = structure.push((0, 1));
        assert_eq!(support, 1);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 0);
        assert_eq!(structure.labels_support().iter().eq([1, 0].iter()), true);

        let support = structure.push((1, 1));
        assert_eq!(structure.is_empty(), true);
        assert_eq!(structure.label_support(0), 0);
        assert_eq!(structure.label_support(1), 0);

        structure.push((2, 1));

        assert_eq!(structure.limit.iter().eq([0, 0, -1, -1].iter()), true);

        structure.backtrack();
        assert_eq!(structure.limit.iter().eq([0, 0, -1].iter()), true);

        structure.backtrack();
        assert_eq!(structure.limit.iter().eq([0, 0].iter()), true);
        assert_eq!(structure.support, 1);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 0);
        assert_eq!(structure.labels_support().iter().eq([1, 0].iter()), true);
    }

    #[test]
    fn check_reset() {
        let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);

        for i in 0..structure.num_attributes() / 4 {
            &mut structure.push((i, 0));
        }

        structure.reset();

        assert_eq!(structure.support(), 812);
        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
    }

    #[test]
    fn test_temp_push() {
        let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);
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
}
