use crate::dataset::data_trait::Dataset;
use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, DoublePointerData, Item, Position, Support};

#[derive(Debug)]
pub struct Part {
    // TODO: Add an Iterator for this part
    // pub(crate) elements : &'elem Vec<usize>,
    pub(crate) begin: usize,
    pub(crate) end: usize,
}

impl Part {
    fn size(&self) -> usize {
        if self.end < self.begin {
            return 0;
        }
        self.end - self.begin + 1
    }
}

struct DoublePointerStructure<'data> {
    input: &'data DoublePointerData,
    elements: Vec<usize>,
    support: Support,
    labels_support: Vec<Support>,
    num_labels: usize,     // TODO: This is not needed
    num_attributes: usize, // TODO: This is not needed
    position: Position,
    // state: Vec<Vec<Part>>, // TODO: The wish here was to have part for each class and/or each value of the attribute
    state: Vec<Part>,
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
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }

        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }
        if let Some(last_state) = self.state.last() {
            for i in last_state.begin..=last_state.end {
                self.labels_support[self.input.target[i]] += 1;
            }
        }
        &self.labels_support
    }

    fn support(&mut self) -> Support {
        if self.support != Support::MAX {
            return self.support;
        }
        self.support = 0;
        if let Some(last_state) = self.state.last() {
            self.support = last_state.size();
        }
        self.support
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
        if !self.position.is_empty() {
            self.position.pop();
            self.state.pop();
            self.support = Support::MAX;
            self.labels_support.clear();
        }
    }

    fn temp_push(&mut self, item: Item) -> Support {
        todo!()
    }

    fn reset(&mut self) {
        self.position = Vec::with_capacity(self.num_attributes);
        self.state = vec![Part {
            begin: 0,
            end: self.input.inputs[0].len() - 1,
        }];
        self.support = self.input.inputs[0].len();
        self.labels_support.clear();
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
        let support = inputs.inputs[0].len();

        let mut state = vec![];

        let part = Part {
            begin: 0,
            end: support - 1,
        };

        state.push(part);

        Self {
            input: inputs,
            elements: (0..support).collect::<Vec<usize>>(),
            support: inputs.inputs.len(),
            labels_support: vec![],
            num_labels: inputs.num_labels,
            num_attributes: inputs.num_attributes,
            position: vec![],
            state,
            last_position: None,
        }
    }

    fn pushing(&mut self, item: Item) {
        // TODO: I will ignore this case for now
        // if self.last_position.is_some() {
        //     let last = self.last_position.take().unwrap();
        //     if last.0 == item.0 {
        //         self.state.push(last.1); // TODO: Check if this is correct I don't really trust this because it is hard to work with the dynamic branching
        //     }
        // } else {
        // }
        self.support = Support::MAX;
        self.labels_support.clear();

        if let Some(last_state) = self.state.last() {
            let inputs = &self.input.inputs;
            let mut begin = last_state.begin;
            let mut end = last_state.end;

            loop {
                while begin < end {
                    if inputs[item.0][self.elements[begin]] == 0 {
                        begin += 1;
                    } else {
                        break;
                    }
                }

                if begin > end + 1 {
                    break;
                }

                while end > begin {
                    if inputs[item.0][self.elements[end]] == 1 {
                        end -= 1;
                    } else {
                        break;
                    }
                }

                if begin == end {
                    break;
                }

                self.elements.swap(begin, end);
                begin += 1;
            }

            let mut part = Part {
                begin: last_state.begin,
                end: begin - 1,
            };
            if item.1 == 1 {
                part = Part {
                    begin: end,
                    end: last_state.end,
                };
            }
            self.support = part.size();
            self.state.push(part);
            self.labels_support();
        }
    }
}

#[cfg(test)]
mod test_double_pointer {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use crate::structures::double_pointer::DoublePointerStructure;
    use crate::structures::structure_trait::Structure;

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

    #[test]
    fn test_simple_part() {
        let dataset = BinaryDataset::load("test_data/small_.txt", false, 0.0);
        let bitset_data = DoublePointerStructure::format_input_data(&dataset);
        let mut structure = DoublePointerStructure::new(&bitset_data);
        // println!("Input: {:?}", structure.input.inputs);
        // println!("Elements: {:?}", structure.elements);
        let item = (0, 0);
        structure.push(item);
        println!("Support: {:?}", structure.support());
        println!("{:?}", structure.state);
        println!("Labels Support: {:?}", structure.labels_support());
        structure.push((1, 1));
        println!("Support: {:?}", structure.support());
        println!("Labels Support: {:?}", structure.labels_support());
        structure.backtrack();
        println!("Support: {:?}", structure.support());
        println!("Labels Support: {:?}", structure.labels_support());

        // structure.push(item);
        println!("Elements: {:?}", structure.elements);

        // assert_eq!(bitset_data.support, 4);
        // assert_eq!(bitset_data.state.len(), 2);
        // assert_eq!(bitset_data.state[0].begin, 0);
        // assert_eq!(bitset_data.state[0].end, 1);
        // assert_eq!(bitset_data.state[1].begin, 2);
        // assert_eq!(bitset_data.state[1].end, 3);
        // assert_eq!(bitset_data.elements, [1, 0, 2, 3]);
        // assert_eq!(bitset_data.position, [item]);
    }
}
