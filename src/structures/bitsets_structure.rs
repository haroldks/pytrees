use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{BitsetStackState, BitsetStructData, Position, Support};

#[derive(Clone)]
struct BitsetStructure<'data> {
    inputs: &'data BitsetStructData,
    support: Support,
    num_labels: usize,
    position: Position,
    state: BitsetStackState,
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
            let target_pos = size - tid - 1;
            let class = data_ref.0[target_pos];
            targets[class][row_chunk] |= 1u64 << (target_pos % 64);
        }

        BitsetStructData { inputs, targets }
    }
}

#[cfg(test)]
mod test_bitsets {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;

    #[test]
    fn load_bitset_data() {
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let bitset_data = BitsetStructure::format_input_data(&dataset);
        println!("{:?}", bitset_data.inputs);
        println!("{:?}", bitset_data.targets);
    }
}
