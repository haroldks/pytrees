use crate::dataset::data_trait::Dataset;
use crate::structures::bitsets_structure::BitsetStructure;
use crate::structures::caching::trie::{Data, DataTrait};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Bitset, BitsetStructData, Support};
use std::cmp::max;
use std::fmt::Debug;

#[derive(Debug, Default)]
pub struct SimilarDatasets<T> {
    pub first: Similarity<T>,
    pub second: Similarity<T>,
}

impl<T> SimilarDatasets<T>
where
    T: DataTrait + Default + Debug,
{
    pub fn new() -> Self {
        Self {
            first: Similarity::default(),
            second: Similarity::default(),
        }
    }

    pub fn update(&mut self, data: &T, structure: &mut RSparseBitsetStructure) -> bool {
        let error = match data.get_node_error() == <usize>::MAX {
            true => data.get_lower_bound(),
            false => data.get_node_error(),
        };

        if error == 0 {
            return false;
        }

        let empty = self.set_empty(data, structure);
        if empty {
            return true;
        }
        let (first_in, first_out) = self.first.difference(structure);
        let (second_in, second_out) = self.second.difference(structure);
        if first_in + first_out < second_in + second_out {
            self.first.update(structure, data);
            true
        } else {
            self.second.update(structure, data);
            true
        }
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_empty() && self.second.is_empty()
    }

    fn set_empty(&mut self, data: &T, structure: &mut RSparseBitsetStructure) -> bool {
        if self.first.is_empty() {
            self.first.update(structure, data);
            return true;
        } else if self.second.is_empty() {
            self.second.update(structure, data);
            return true;
        }
        false
    }

    pub fn compute_similarity(&mut self, structure: &mut RSparseBitsetStructure) -> usize {
        let mut bound = 0;
        let saved = [&self.first, &self.second];
        for similarity in saved {
            if !similarity.is_empty() {
                let diff = structure.difference(similarity, false);
                bound = max(bound, similarity.error.saturating_sub(diff));
            }
        }
        bound
    }
}

#[derive(Default, Debug)]
pub struct Similarity<T> {
    pub state: Bitset,
    pub limit: isize,
    pub index: Vec<usize>,
    pub error: usize,
    pub support: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Similarity<T>
where
    T: DataTrait + Default + Default,
{
    pub fn new() -> Self {
        Self {
            state: Bitset::new(),
            index: vec![],
            limit: 0,
            error: 0,
            support: 0,
            _phantom: Default::default(),
        }
    }

    pub fn update(&mut self, structure: &mut RSparseBitsetStructure, data: &T) {
        self.state = structure.get_last_state_bitset();
        self.index = structure.get_current_index();
        self.limit = structure.get_current_limit();
        self.error = data.get_node_error();
        self.support = structure.get_support();
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub fn difference(&self, structure: &mut RSparseBitsetStructure) -> (usize, usize) {
        let (in_cout, out_count) = (
            structure.difference(self, true),
            structure.difference(self, false),
        );
        (in_cout, out_count)
    }
}

#[cfg(test)]
mod slb_tests {
    use crate::algorithms::dl85::DL85;
    use crate::algorithms::dl85_utils::slb::{SimilarDatasets, Similarity};
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::caching::trie::{Data, DataTrait};
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
    use crate::structures::structure_trait::Structure;

    #[test]
    fn test_slb() {
        let dataset = BinaryDataset::load("test_data/anneal.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = RSparseBitsetStructure::new(&bitset_data);

        let mut slb: SimilarDatasets<Data> = SimilarDatasets::new();
        assert_eq!(slb.is_empty(), true);

        let mut data = Data::new();
        data.error = 1;

        structure.push((0, 0));
        slb.update(&data, &mut structure);

        structure.backtrack();

        structure.push((1, 0));

        let elem_in = structure.difference(&slb.first, true);
        let elem_out = structure.difference(&slb.first, false);

        assert_eq!(elem_in, 812);
        assert_eq!(elem_out, 0);
    }
}
