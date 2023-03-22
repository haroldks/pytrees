use crate::dataset::data_types::Data;
use ndarray::{Array, IxDyn};
use std::fs::File;
use std::io::{BufRead, BufReader, Error};

pub trait Dataset {
    fn load(filename: &str, shuffle: bool, split: f64) -> Self;

    fn load_from_numpy(input: &Array<usize, IxDyn>, target: &Array<usize, IxDyn>) -> Self;

    fn size(&self) -> usize;

    fn train_size(&self) -> usize;

    fn num_labels(&self) -> usize;

    fn num_attributes(&self) -> usize;

    fn get_train(&self) -> &Data;

    fn open_file(filename: &str) -> Result<Vec<String>, Error> {
        let input = File::open(filename)?; //Error Handling for missing filename
        let buffered = BufReader::new(input); // Buffer for the file
        Ok(buffered
            .lines()
            .map(|x| x.unwrap())
            .collect::<Vec<String>>())
    }
}
