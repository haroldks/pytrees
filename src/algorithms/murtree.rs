use crate::structures::structure_trait::Structure;

pub struct MurTree {}

impl MurTree {
    fn build_matrix<T>(structure: &mut T) -> Vec<Vec<(usize, usize)>>
    where
        T: Structure,
    {
        let num_features = structure.num_attributes();
        let mut matrix = vec![vec![(0usize, 0usize); num_features]; num_features];
        for i in 0..num_features {
            structure.push((i, 1));
            let val = structure.labels_support();
            matrix[i][i] = (val[0], val[1]);

            for j in i + 1..num_features {
                structure.push((j, 1));
                let val = structure.labels_support();
                structure.backtrack();
                matrix[i][j] = (val[0], val[1]);
                matrix[j][i] = (val[0], val[1]);
            }
            structure.backtrack();
        }
        matrix
    }
}

#[cfg(test)]
mod murtree_test {
    use crate::algorithms::murtree::MurTree;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::bitsets_structure::BitsetStructure;
    use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
    use crate::structures::structures_types::BitsetStructData;

    #[test]
    fn test_full_data_matrix_building() {
        // TODO: fix when there is only one chunk issue

        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
        let mut structure = BitsetStructure::new(&bitset_data);

        let matrix = MurTree::build_matrix(&mut structure);
    }
}
