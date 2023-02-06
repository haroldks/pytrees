use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, DoublePointerData, Item, Position, Support};

pub struct Part {
    // TODO: Add an Iterator for this part
    // pub(crate) elements : &'elem Vec<usize>,
    pub(crate) begin: usize,
    pub(crate) end: usize,
}

impl Part {
    fn size(&self) -> usize {
        self.end - self.begin + 1
    }
}

struct DoublePointerStructure<'data> {
    input: &'data DoublePointerData,
    elements: Vec<usize>,
    support: Support,
    num_labels: usize,
    num_attributes: usize,
    position: Position,
    state: Vec<Vec<Part>>,
    last_position: Option<(Attribute, Vec<Part>)>,
}

impl<'data> Structure for DoublePointerStructure<'data> {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> Support {
        todo!()
    }

    fn labels_support(&mut self) -> &[Support] {
        todo!()
    }

    fn support(&mut self) -> Support {
        let mut support = 0;
        if self.position.is_empty() {
            return self.elements.len();
        } else if !self.state.is_empty() {
            if let Some(last_position) = self.position.last() {
                let part_idx = last_position.1;
                if let Some(part) = self.state.last() {
                    return part[part_idx].size();
                }
            }
        }
        support
    }

    fn get_support(&self) -> Support {
        self.support
    }

    fn push(&mut self, item: Item) -> Support {
        self.position.push(item);
        self.pushing(item);
        self.support
    }

    fn backtrack(&mut self) {
        todo!()
    }

    fn temp_push(&mut self, item: Item) -> Support {
        todo!()
    }

    fn reset(&mut self) {
        self.position = vec![];
        self.state = vec![];
    }

    fn get_position(&self) -> &Position {
        &self.position
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

    pub fn new(inputs: &'data DoublePointerData) -> Self {
        let support = inputs.inputs.len();

        let mut initial_state = vec![];
        for _ in 0..2 {
            let part = Part {
                begin: 0,
                end: support,
            };
            initial_state.push(part)
        }

        Self {
            input: inputs,
            elements: (0..inputs.inputs.len()).collect::<Vec<usize>>(),
            support: inputs.inputs.len(),
            num_labels: inputs.num_labels,
            num_attributes: inputs.num_attributes,
            position: vec![],
            state: vec![],
            last_position: None,
        }
    }

    fn pushing(&mut self, item: Item) {
        if self.last_position.is_some() {
            let last = self.last_position.take().unwrap();
            if last.0 == item.0 {
                self.state.push(last.1);
            }
        } else {
        }
    }
}

#[cfg(test)]
mod test_double_pointer {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::double_pointer::DoublePointerStructure;

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
}
