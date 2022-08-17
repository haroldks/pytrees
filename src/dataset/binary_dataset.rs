use self::super::data_trait::Dataset;
use super::data_types::Data;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::{BufRead, BufReader, Error};

struct BinaryDataset {
    filename: String,
    shuffle: bool,
    split: f64,
    train: Data,
    test: Option<Data>,
    size: usize,
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
            true => Some(data.drain(0..test_size).collect::<Vec<String>>()),
            false => None,
        };

        let train = data.drain(test_size..).collect::<Vec<String>>();

        Self {
            filename: filename.to_string(),
            shuffle,
            split,
            train,
            test,
            size,
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn partition_size(&self, label: usize) -> usize {
        todo!()
    }

    fn num_labels(&self) -> usize {
        todo!()
    }
}

#[cfg(test)]
mod test_binary_dataset {
    use crate::dataset::binary_dataset::BinaryDataset;
    use crate::dataset::data_trait::Dataset;
    use std::io::Error;
    use std::panic;

    #[test]
    fn can_open_file() {
        let dataset = BinaryDataset::open_file("datasets/small.txt");

        let dataset = match dataset {
            Ok(file) => file,
            Err(error) => {
                panic!("Should not panic")
            }
        };
    }

    #[test]
    #[should_panic(expected = "Missing File")]
    fn missing_file() {
        let dataset = BinaryDataset::open_file("datasets/missing.txt");

        let dataset = match dataset {
            Ok(file) => file,
            Err(error) => {
                panic!("Missing File")
            }
        };
    }

    #[test]
    fn data_is_retrieved() {
        let dataset = BinaryDataset::open_file("datasets/small.txt");
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
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.0);
        assert_eq!(dataset.test.is_none(), true);
    }

    #[test]
    fn binary_dataset_no_shuffle_and_half_split() {
        let dataset = BinaryDataset::load("datasets/small.txt", false, 0.5);
        assert_eq!(dataset.test.is_some(), true);
        let data_ref = dataset.test.as_ref().unwrap();
        assert_eq!(data_ref.len(), dataset.size / 2);
        let content = vec!["0 1 0 1", "0 0 1 1"];
        assert_eq!(data_ref.iter().eq(content.iter()), true);
    }
}
