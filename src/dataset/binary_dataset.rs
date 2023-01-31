use self::super::data_trait::Dataset;
use super::data_types::Data;
use ndarray::{Array, IxDyn};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashSet;

pub struct BinaryDataset {
    filename: String,
    shuffle: bool,
    split: f64,
    train: Data,
    test: Option<Data>,
    size: usize,
    train_size: usize,
    num_labels: usize,
    num_attributes: usize,
}

impl Dataset for BinaryDataset {
    fn load(filename: &str, shuffle: bool, split: f64) -> Self {
        let mut data = Self::open_file(filename).unwrap();
        let size = data.len();

        if shuffle {
            data.shuffle(&mut thread_rng())
        }

        let test_size = (size as f64 * split) as usize;

        let test = match test_size >= 1 {
            true => Some(BinaryDataset::create_set(
                data.drain(0..test_size).collect::<Vec<String>>(),
            )),
            false => None,
        };

        let train = BinaryDataset::create_set(data);
        let train_size = train.0.len();
        let num_attributes = train.1[0].len();
        let num_labels = train.0.iter().collect::<HashSet<_>>().len();

        Self {
            filename: filename.to_string(),
            shuffle,
            split,
            train,
            test,
            size,
            train_size,
            num_labels,
            num_attributes,
        }
    }

    fn load_from_numpy(input: &Array<usize, IxDyn>, target: &Array<usize, IxDyn>) -> Self {
        let targets = target.to_owned().into_raw_vec();
        let mut inputs = vec![];
        for row in input.rows() {
            inputs.push(row.to_vec());
        }
        let train_size = inputs.len();
        let num_attributes = inputs[0].len();
        let num_labels = targets.iter().collect::<HashSet<_>>().len();
        let train: Data = (targets, inputs);

        Self {
            filename: "from_python".to_string(),
            shuffle: false,
            split: 0.0f64,
            train,
            test: None,
            size: train_size,
            train_size,
            num_labels,
            num_attributes,
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn train_size(&self) -> usize {
        self.train_size
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn get_train(&self) -> &Data {
        &self.train
    }
}

impl BinaryDataset {
    fn create_set(data: Vec<String>) -> Data {
        let data = data
            .iter()
            .map(|line| {
                line.split_whitespace()
                    .map(|y| y.parse().unwrap())
                    .collect::<Vec<usize>>()
            })
            .collect::<Vec<Vec<usize>>>();
        let targets = data.iter().map(|row| row[0]).collect::<Vec<usize>>();
        let rows = data
            .iter()
            .map(|row| row[1..].to_vec())
            .collect::<Vec<Vec<usize>>>();
        (targets, rows)
    }
}

#[cfg(test)]
mod test_binary_dataset {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use ndarray::{arr1, arr2, Array1, Array2};
    use numpy::pyo3::ffi::binaryfunc;
    use std::panic;

    #[test]
    fn can_open_file() {
        let dataset = BinaryDataset::open_file("test_data/small.txt");

        let _dataset = match dataset {
            Ok(file) => file,
            Err(_error) => {
                panic!("Should not panic")
            }
        };
    }

    #[test]
    #[should_panic(expected = "Missing File")]
    fn missing_file() {
        let dataset = BinaryDataset::open_file("test_data/missing.txt");

        let _dataset = match dataset {
            Ok(file) => file,
            Err(_error) => {
                panic!("Missing File")
            }
        };
    }

    #[test]
    fn data_is_retrieved() {
        let dataset = BinaryDataset::open_file("test_data/small.txt");
        let content = vec!["0 1 0 1", "0 0 1 1", "1 0 0 0", "1 0 1 0"];

        let dataset = match dataset {
            Ok(file) => file,
            Err(_) => {
                panic!("Should not panic")
            }
        };
        assert_eq!(dataset.iter().eq(content.iter()), true);
    }

    #[test]
    fn binary_dataset_no_shuffle_and_no_split() {
        let dataset = BinaryDataset::load("test_data/small.txt", false, 0.0);
        assert_eq!(dataset.filename, "test_data/small.txt");
        assert_eq!(dataset.shuffle, false);
        assert_eq!(dataset.test.is_none(), true);
    }

    #[test]
    fn binary_dataset_no_shuffle_and_half_split() {
        let mut dataset = BinaryDataset::load("test_data/small.txt", false, 0.5);
        assert_eq!(dataset.test.is_some(), true);
        let data = dataset.test.take().unwrap();
        let rows = data.1;
        assert_eq!(data.0.len(), dataset.size() / 2);
        let content = vec![vec![1, 0, 1], vec![0, 1, 1]];
        assert_eq!(rows.iter().eq(content.iter()), true);
    }

    #[test]
    fn binary_dataset_shuffled_and_half_split() {
        let mut dataset = BinaryDataset::load("test_data/small.txt", true, 0.25);
        assert_eq!(dataset.test.is_some(), true);
        let data = dataset.test.take().unwrap();
        assert_eq!(data.0.len(), 1);
    }

    #[test]
    fn binary_dataset_size_and_label() {
        let dataset = BinaryDataset::load("test_data/small.txt", true, 0.0);
        assert_eq!(dataset.size(), 4);
        assert_eq!(dataset.num_labels(), 2);
    }

    #[test]
    fn binary_dataset_numpy() {
        let targets = arr1(&[0usize, 0, 1, 1]).into_dyn();
        let input = arr2(&[[1usize, 0, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0]]).into_dyn();
        let dataset = BinaryDataset::load_from_numpy(&input, &targets);

        assert_eq!(dataset.size(), 4);
        assert_eq!(dataset.num_labels(), 2);
        assert_eq!(dataset.shuffle, false);
        assert_eq!(dataset.test.is_none(), true);
    }
}
