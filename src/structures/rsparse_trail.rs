use crate::algorithms::dl85_utils::slb::Similarity;
use crate::dataset::data_trait::Dataset;
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::caching::trie::DataTrait;
use crate::structures::structure_trait::{IsTrail, Structure};
use crate::structures::structures_types::{
    Bitset, BitsetStackState, BitsetStructData, Item, Position, Support,
};
use search_trail::{ReversibleU64, SaveAndRestore, StateManager, U64Manager};

impl<'data> IsTrail for RSparseTrail<'data> {}

pub struct RSparseTrail<'data> {
    inputs: &'data BitsetStructData,
    support: Support,
    labels_support: Vec<Support>,
    num_labels: usize,
    num_attributes: usize,
    position: Position,
    state_manager: StateManager,
    state: Vec<ReversibleU64>,
    index: Vec<usize>,
    limit: Vec<isize>,
    distance: ReversibleU64, // Steps to restore to attain the initial state
}

impl<'data> Structure for RSparseTrail<'data> {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        // FIXME: Useless
        let state = &self.state;
        let support = Support::MAX;

        if label < self.num_labels {
            if let Some(limit) = self.limit.last() {
                let mut count = 0;
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        let val = self.state_manager.get_u64(state[cursor]);
                        count += (label_bitset[cursor] & val).count_ones()
                    }
                }
                return count as Support;
            }
        }
        support
    }

    fn labels_support(&mut self) -> &[Support] {
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }

        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }

        if let Some(limit) = self.limit.last() {
            if self.num_labels == 2 {
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[0];
                    let mut count = 0;
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        let val = self.state_manager.get_u64(self.state[cursor]);
                        count += (label_bitset[cursor] & val).count_ones()
                    }
                    self.labels_support[0] = count as Support;
                    self.labels_support[1] = self.support() - count as Support;
                }
                return &self.labels_support;
            }

            for label in 0..self.num_labels {
                let mut count = 0;
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        let val = self.state_manager.get_u64(self.state[cursor]);
                        let word = label_bitset[cursor] & val;
                        count += (label_bitset[cursor] & val).count_ones()
                    }
                }
                self.labels_support[label] = count as Support;
            }
            return &self.labels_support;
        }
        &self.labels_support
    }

    fn support(&mut self) -> Support {
        if !self.support == Support::MAX {
            return self.support;
        }
        self.support = 0;
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                for i in 0..(*limit + 1) as usize {
                    let cursor = self.index[i];
                    let val = self.state_manager.get_u64(self.state[cursor]);
                    self.support += val.count_ones() as Support;
                }
            }
        }
        self.support
    }

    fn get_support(&self) -> Support {
        self.support
    }

    fn push(&mut self, item: Item) -> Support {
        self.position.push(item);
        let current_distance = self.state_manager.get_u64(self.distance);
        self.state_manager
            .set_u64(self.distance, current_distance + 1);
        self.state_manager.save_state();
        self.pushing(item);

        self.support()
    }

    fn backtrack(&mut self) {
        // TODO: Remove the support computation
        if !self.position.is_empty() {
            self.position.pop();
            if self.is_empty() {
                self.limit.pop();
            } else if let Some(limit) = self.limit.last() {
                self.state_manager.restore_state();
                self.limit.pop();
            }
            self.support = Support::MAX;
            self.labels_support.clear();
            // self.support();
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        // TODO: Change this to avoid recomputing the support & labels support
        let mut support = 0;
        if let Some(limit) = self.limit.last() {
            let mut limit = *limit;
            if limit >= 0 {
                let feature_vec = &self.inputs.inputs[item.0];
                let mut lim = limit as usize;
                for i in (0..lim + 1).rev() {
                    let cursor = self.index[i];
                    let val = self.state_manager.get_u64(self.state[cursor]);
                    let word = match item.1 {
                        0 => val & !feature_vec[cursor],
                        _ => val & feature_vec[cursor],
                    };
                    let word_count = word.count_ones() as Support;
                    support += word_count;
                }
            }
        }
        support
    }

    fn reset(&mut self) {
        self.position = Vec::with_capacity(self.num_attributes);
        self.limit = Vec::with_capacity(self.num_attributes);
        self.limit.push((self.inputs.chunks - 1) as isize);
        let distance = self.state_manager.get_u64(self.distance);
        for _ in 0..distance + 1 {
            self.state_manager.restore_state();
        }
        self.support = self.inputs.size as Support;
        self.labels_support.clear();
    }

    fn get_position(&self) -> &Position {
        &self.position
    }
}

impl<'data> RSparseTrail<'data> {
    pub fn format_input_data<T>(data: &T) -> BitsetStructData
    where
        T: Dataset,
    {
        BitsetStructure::format_input_data(data)
    }

    pub fn new(inputs: &'data BitsetStructData) -> RSparseTrail<'data> {
        let index = (0..inputs.chunks).collect::<Vec<usize>>();
        let num_attributes = inputs.inputs.len();
        let mut state = Vec::with_capacity(inputs.chunks);
        let mut manager = StateManager::default();
        for i in 0..inputs.chunks {
            let n = manager.manage_u64(u64::MAX);
            state.push(n);
        }

        if inputs.size % 64 != 0 {
            let first_dead_bit = 64 - (inputs.chunks * 64 - inputs.size);
            let first_chunk = &mut state[0];

            let mut val = manager.get_u64(*first_chunk);
            for i in (first_dead_bit..64).rev() {
                let int_mask = 1u64 << i;
                val &= !int_mask;
            }
            manager.set_u64(*first_chunk, val);
        }

        let distance = manager.manage_u64(0);

        manager.save_state(); // Save the initial state of the manager

        let mut limit = Vec::with_capacity(num_attributes);
        limit.push((inputs.chunks - 1) as isize);

        let mut structure = RSparseTrail {
            inputs,
            support: inputs.size as Support,
            labels_support: Vec::with_capacity(inputs.targets.len()),
            num_labels: inputs.targets.len(),
            num_attributes,
            position: Vec::with_capacity(num_attributes),
            state_manager: manager,
            state,
            index,
            limit,
            distance,
        };
        structure.support();
        structure
    }

    fn pushing(&mut self, item: Item) {
        self.support = 0;
        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }

        if let Some(limit) = self.limit.last() {
            let mut limit = *limit;
            if limit >= 0 {
                let feature_vec = &self.inputs.inputs[item.0];
                let mut lim = limit as usize;
                for i in (0..lim + 1).rev() {
                    let cursor = self.index[i];
                    let val = self.state_manager.get_u64(self.state[cursor]);
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
                        let word_count = word.count_ones() as Support;
                        self.support += word_count;
                        if self.num_labels == 2 {
                            let label_val = &self.inputs.targets[0][cursor];
                            let zero_count = (label_val & word).count_ones() as Support;
                            self.labels_support[0] += zero_count;
                            self.labels_support[1] += (word_count - zero_count);
                        } else {
                            for j in 0..self.num_labels {
                                let label_val = &self.inputs.targets[j][cursor];
                                self.labels_support[j] +=
                                    (label_val & word).count_ones() as Support;
                            }
                        }

                        self.state_manager.set_u64(self.state[cursor], word);
                    }
                }
            }

            self.limit.push(limit);
        }
    }

    pub fn get_last_state_bitset(&self) -> Bitset {
        let state = &self.state;
        let mut to_export: Bitset = vec![0; self.inputs.chunks];
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                for i in 0..(*limit + 1) {
                    let cursor = self.index[i as usize];
                    let val = self.state_manager.get_u64(state[cursor]);
                    to_export[cursor] = val;
                }
            }
        }
        to_export
    }

    fn is_empty(&self) -> bool {
        if let Some(limit) = self.limit.last() {
            return *limit < 0;
        }
        false
    }

    pub fn get_current_index(&self) -> Vec<usize> {
        self.index.clone()
    }

    pub fn get_current_limit(&self) -> isize {
        self.limit.last().copied().unwrap_or(-1)
    }

    pub fn difference<T: DataTrait>(&self, similarity: &Similarity<T>, data_in: bool) -> usize {
        let struc_limit = self.get_current_limit();

        let limit = match data_in {
            true => struc_limit,
            false => similarity.limit,
        };

        let index = match data_in {
            true => &self.index,
            false => &similarity.index,
        };
        let mut count = 0;
        if limit >= 0 {
            for cursor in index.iter().take(limit as usize + 1) {
                let val = match struc_limit == -1 {
                    true => 0,
                    false => self.state_manager.get_u64(self.state[*cursor]),
                };
                let diff = match data_in {
                    true => val & !similarity.state[*cursor],
                    false => similarity.state[*cursor] & !val,
                };
                count += diff.count_ones();
            }
            return count as usize;
        }
        0
    }
}

#[cfg(test)]
mod test_trail {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::rsparse_trail::RSparseTrail;
    use crate::structures::structure_trait::Structure;
    use search_trail::{ReversibleU64, SaveAndRestore, StateManager, U64Manager};
    use std::panic;

    #[test]
    fn test() {
        let mut manager = StateManager::default();
        let n = manager.manage_u64(u64::MAX);
        let distance = manager.manage_u64(0);
        manager.save_state();

        manager.set_u64(n, 0);

        let val = manager.get_u64(distance);
        manager.set_u64(distance, val + 1);

        manager.save_state();
        manager.set_u64(n, 1);
        let val = manager.get_u64(distance);
        manager.set_u64(distance, val + 1);
        manager.save_state();

        for _ in 0..manager.get_u64(distance) + 1 {
            manager.restore_state();
            println!("{}", manager.get_u64(n));
        }
        println!("{}", manager.get_u64(distance));
    }

    #[test]
    fn test_trail_stats() {
        let dataset = BinaryDataset::load("test_data/rsparse_dataset.txt", false, 0.0);
        let bitset_data = RSparseTrail::format_input_data(&dataset);
        let mut structure = RSparseTrail::new(&bitset_data);

        let expected_support = 192;
        assert_eq!(structure.support, expected_support);

        let expected_label_supports = [64usize, 128];
        assert_eq!(structure.label_support(0), expected_label_supports[0]);
        assert_eq!(structure.label_support(1), expected_label_supports[1]);
    }

    #[test]
    fn test_trail_branching_in_data() {
        let dataset = BinaryDataset::load("test_data/rsparse_dataset.txt", false, 0.0);
        let bitset_data = RSparseTrail::format_input_data(&dataset);
        let mut structure = RSparseTrail::new(&bitset_data);

        let item = (0, 1);
        let support = structure.push(item);

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

        assert_eq!(structure.support(), 128);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 1);
        }

        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 128);
        assert_eq!(structure.label_support(0), 0);
    }

    #[test]
    fn compute_state_on_small_dataset_with_trail() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        let bitset_data = RSparseTrail::format_input_data(&dataset);
        let mut structure = RSparseTrail::new(&bitset_data);
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
        assert_eq!(structure.support(), 1);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 0);
        assert_eq!(structure.labels_support().iter().eq([1, 0].iter()), true);
    }

    #[test]
    fn test_temp_push_on_trail() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = RSparseTrail::format_input_data(&dataset);
        let mut structure = RSparseTrail::new(&bitset_data);
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
    fn check_trail_reset() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = RSparseTrail::format_input_data(&dataset);
        let mut structure = RSparseTrail::new(&bitset_data);

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
    fn testq() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = RSparseTrail::format_input_data(&dataset);
        let mut structure = RSparseTrail::new(&bitset_data);

        let support = structure.push((4, 0));
        println!("support: {}", support);
        println!("Labels support: {:?}", structure.labels_support());
        let new_support = structure.push((5, 0));
        println!("new_support*: {}", new_support);
        println!("Labels support: {:?}", structure.labels_support());
        for trail in structure.state.iter() {
            println!(
                "before trail: {:?}",
                structure.state_manager.get_u64(*trail)
            );
        }
        println!("");

        structure.backtrack();

        for trail in structure.state.iter() {
            println!("after trail: {:?}", structure.state_manager.get_u64(*trail));
        }
        println!("");

        println!("Positions: {:?}", structure.get_position());
        println!("support: {}", structure.support());
        println!("Labels support: {:?}", structure.labels_support());
        let new_support = structure.push((5, 1));
        println!("new_support*: {}", new_support);
        println!("Labels support: {:?}", structure.labels_support());
    }
}
