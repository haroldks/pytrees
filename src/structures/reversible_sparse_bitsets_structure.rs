use crate::algorithms::dl85_utils::slb::Similarity;
use crate::dataset::data_trait::Dataset;
use crate::structures::binary_tree::{NodeData, Tree, TreeNode};
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::caching::trie::DataTrait;
use crate::structures::structure_trait::{BitsetTrait, Structure};
use crate::structures::structures_types::{
    Bitset, BitsetStackState, BitsetStructData, Index, Item, LeafInfo, Position, StateCollection,
    Support,
};

#[derive(Clone)]
pub struct RSparseBitsetStructure<'data> {
    inputs: &'data BitsetStructData,
    support: Support,
    labels_support: Vec<Support>,
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
                        if let Some(val) = state[cursor].last() {
                            count += (label_bitset[cursor] & val).count_ones()
                        }
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

        let state = &self.state;

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
                        if let Some(val) = state[cursor].last() {
                            count += (label_bitset[cursor] & val).count_ones()
                        }
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
                        if let Some(val) = state[cursor].last() {
                            let word = label_bitset[cursor] & val;
                            count += (label_bitset[cursor] & val).count_ones()
                        }
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
        let state = &self.state;
        self.support = 0;
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                for i in 0..(*limit + 1) as usize {
                    let cursor = self.index[i];
                    if let Some(val) = state[cursor].last() {
                        self.support += val.count_ones() as Support;
                    }
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
                for i in 0..(*limit + 1) as usize {
                    self.state[self.index[i]].pop();
                }
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
                    if let Some(val) = self.state[cursor].last() {
                        let word = match item.1 {
                            0 => val & !feature_vec[cursor],
                            _ => val & feature_vec[cursor],
                        };
                        let word_count = word.count_ones() as Support;
                        support += word_count;
                    }
                }
            }
        }
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
        self.support = self.inputs.size as Support;
        self.labels_support.clear();
    }
    fn get_position(&self) -> &Position {
        &self.position
    }
}

impl<'data> BitsetTrait for RSparseBitsetStructure<'data> {
    fn extract_leaf_bitvector(
        &mut self,
        tree: &Tree<NodeData>,
        index: Index,
        position: &mut Vec<Item>,
        collector: &mut Vec<LeafInfo>,
    ) {
        let mut left_index = 0;
        let mut right_index = 0;
        let mut attribute = None;
        if let Some(node) = tree.get_node(index) {
            left_index = node.left;
            right_index = node.right;
            attribute = node.value.test;
        }

        if left_index == right_index {
            let mut error = <usize>::MAX;
            if let Some(node) = tree.get_node(index) {
                error = node.value.error;
            }

            // Is leaf
            collector.push(LeafInfo {
                index,
                position: position.clone(),
                bitset: self.get_last_state_bitset(),
                error,
            }); // Bizarre ca
                // position.pop();
        }

        if left_index > 0 {
            if let Some(left_node) = tree.get_node(left_index) {
                let item = (attribute.unwrap(), 0);
                position.push(item);
                self.push(item);
                self.extract_leaf_bitvector(tree, left_index, position, collector);
                position.pop();
                self.backtrack()
            }
        }

        if right_index > 0 {
            if let Some(right_node) = tree.get_node(right_index) {
                let item = (attribute.unwrap(), 1);
                position.push(item);
                self.push(item);
                self.extract_leaf_bitvector(tree, right_index, position, collector);
                position.pop();
                self.backtrack()
            }
        }
    }

    fn set_state(&mut self, state: &Bitset, position: &Position) {
        self.position = position.clone();
        let mut new_state = vec![Vec::with_capacity(self.num_attributes); self.inputs.chunks];
        for (i, s) in new_state.iter_mut().enumerate() {
            s.push(state[i]);
        }
        self.state = new_state;
        self.support = Support::MAX;
        self.labels_support.clear();
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
        for s in state.iter_mut().take(inputs.chunks) {
            s.push(u64::MAX);
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
            support: inputs.size as Support,
            labels_support: Vec::with_capacity(inputs.targets.len()),
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

    fn is_empty(&self) -> bool {
        if let Some(limit) = self.limit.last() {
            return *limit < 0;
        }
        false
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

                            self.state[cursor].push(word);
                        }
                    }
                }
            }

            self.limit.push(limit)
        }
    }

    // Start : Methods to evaluate the structure fo similarity lower bound

    pub fn get_last_state_bitset(&self) -> Bitset {
        let state = &self.state;
        let mut to_export: Bitset = vec![0; self.inputs.chunks];
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                for i in 0..(*limit + 1) {
                    let cursor = self.index[i as usize];
                    if let Some(val) = state[cursor].last() {
                        to_export[cursor] = *val;
                    }
                }
            }
        }
        to_export
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
                    false => *self.state[*cursor].last().unwrap_or(&0),
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
    // End : Methods to evaluate the structure fo similarity lower bound
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
        let dataset = BinaryDataset::load("test_data/rsparse_dataset.txt", false, 0.0);
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
        let dataset = BinaryDataset::load("test_data/rsparse_dataset.txt", false, 0.0);
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
        let dataset = BinaryDataset::load("test_data/rsparse_dataset.txt", false, 0.0);
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

        assert_eq!(structure.support(), 128);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 1);
        }

        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 128);
        assert_eq!(structure.label_support(0), 0);
    }

    #[test]
    fn compute_state_on_small_dataset() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
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
        assert_eq!(structure.support(), 1);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 0);
        assert_eq!(structure.labels_support().iter().eq([1, 0].iter()), true);
    }

    #[test]
    fn check_reset() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
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
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
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
