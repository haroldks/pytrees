use crate::algorithms::algorithm_trait::Algorithm;
use crate::structures::binary_tree::{NodeData, Tree};
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Support};

enum Heuristic {
    InformationGain,
    InformationGainRatio,
    GiniIndex,
    NodeError,
    None,
}

struct LGDT {
    tree: Option<Tree<NodeData<usize>>>,
    error: Option<usize>,
}

impl LGDT {
    pub fn new() -> Self {
        LGDT {
            tree: None,
            error: None,
        }
    }

    pub fn fit<S, F>(structure: &mut S, minsup: Support, maxdepth: Depth, fit_method: F)
    where
        S: Structure,
        F: Fn(&mut S, Support, Depth) -> Tree<NodeData<usize>>,
    {
        let a = fit_method(structure, minsup, maxdepth);
        a.print();
    }
}

#[cfg(test)]
mod lgdt_test {
    use crate::algorithms::algorithm_trait::Algorithm;
    use crate::algorithms::lgdt::LGDT;
    use crate::algorithms::murtree::MurTree;
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::horizontal_binary_structure::HorizontalBinaryStructure;

    #[test]
    fn random() {
        // TODO: fix when there is only one chunk issue
        let dataset = BinaryDataset::load("datasets/anneal.txt", false, 0.0);
        let bitset_data = HorizontalBinaryStructure::format_input_data(&dataset);
        let mut structure = HorizontalBinaryStructure::new(&bitset_data);

        LGDT::fit(&mut structure, 1, 1, MurTree::fit);
    }
}
