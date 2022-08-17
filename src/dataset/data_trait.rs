use std::fs::File;
use std::io::{BufRead, BufReader, Error};

pub trait Dataset {
    fn load(filename: &str, shuffle: bool, split: f64) -> Self;

    fn size(&self) -> usize;

    fn partition_size(&self, label: usize) -> usize;

    fn num_labels(&self) -> usize;

    fn open_file(filename: &str) -> Result<Vec<String>, Error> {
        let input = File::open(&filename)?; //Error Handling for missing filename
        let buffered = BufReader::new(input); // Buffer for the file
        Ok(buffered.lines().map(|x| x.unwrap()).collect::<Vec<String>>())
    }
}
