#![allow(unused)]

use crate::algorithms::algorithm_trait::Algorithm;
use crate::algorithms::dl85::DL85;
use crate::algorithms::dl85_utils::structs_enums::Specialization::Murtree;
use crate::algorithms::dl85_utils::structs_enums::{
    BranchingType, CacheInit, DiscrepancyStrategy, LowerBoundHeuristic, Specialization,
};
use crate::algorithms::idk::IDK;
use crate::algorithms::info_gain::InfoGain;
use crate::algorithms::lds_dl85::LDSDL85;
use crate::algorithms::lgdt::LGDT;
use crate::algorithms::murtree::MurTree;
use crate::dataset::binary_dataset::BinaryDataset;
use crate::dataset::data_trait::Dataset;
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::structures::caching::trie::{Data, TrieNode};
use crate::structures::reversible_sparse_bitsets_structure::RSparseBitsetStructure;
use crate::structures::structure_trait::Structure;
use itertools::Itertools;
use ndarray::s;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
// use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use rayon::prelude::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json;
use serde_json::to_writer;
use std::fs::File;
use std::io::Error;

mod algorithms;
mod dataset;
mod heuristics;
mod post_process;
mod structures;

#[derive(Debug, Serialize, Deserialize)]
struct ExpeRes {
    size: Vec<usize>,
    res: Vec<Vec<f64>>,
}

impl ExpeRes {
    pub fn to_json(&self, filename: String) -> Result<(), Error> {
        if let Err(e) = to_writer(&File::create(filename)?, &self) {
            println!("File Creating error: {}", e.to_string());
        };
        Ok(())
    }
}

fn gen_random_vec(size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut a = vec![];
    for _ in 0..size {
        a.push(rng.gen_range(0..1_000_000) as usize)
    }
    a
}

fn compare(a: &[usize], b: &[usize], n_threads: usize) -> usize {
    let mut rng = rand::thread_rng();

    let size = a.len();

    if n_threads <= 1 {
        let mut value = 0;
        for i in 0..size {
            value += (a[i] & b[i]).count_ones();
        }
        return value as usize;
    }

    let chunk_size = size / n_threads;
    let mut value = 0;
    rayon::scope(|s| {
        let val = Arc::new(Mutex::new(&mut value));
        for i in 0..n_threads {
            // let chunk = &indexes[chunk_start..chunk_end];
            let thread_val = val.clone();
            let res = s.spawn(move |_| {
                let chunk_start = i * chunk_size;
                let chunk_end = if i == n_threads - 1 {
                    size
                } else {
                    (i + 1) * chunk_size
                };

                let mut val = thread_val.lock().unwrap();
                let mut count = 0;
                for idx in chunk_start..chunk_end {
                    count += (a[idx] & b[idx]).count_ones();
                }
                **val += count;
            });
        }
    });
    return value as usize;
}

fn compare_free_contention(a: &[usize], b: &[usize], n_threads: usize) -> usize {
    let size = a.len();

    if n_threads <= 1 {
        let mut value = 0;
        for i in 0..size {
            value += (a[i] & b[i]).count_ones();
        }
        return value as usize;
    }

    let chunk_size = size / n_threads;
    let mut values = vec![0; n_threads];

    values.par_iter_mut().enumerate().for_each(|(i, val)| {
        let chunk_start = i * chunk_size;
        let chunk_end = if i == n_threads - 1 {
            size
        } else {
            (i + 1) * chunk_size
        };

        let mut count = 0;
        for idx in chunk_start..chunk_end {
            count += (a[idx] & b[idx]).count_ones();
        }
        *val = count;
    });

    let mut value = 0;
    for val in values.iter() {
        value += *val;
    }

    value as usize
}

fn main() {
    let dataset = BinaryDataset::load("test_data/ionosphere.txt", false, 0.0);
    // let dataset = BinaryDataset::load(
    //     "experiments/data/parallel_datasets/250_1000000.csv",
    //     false,
    //     0.0,
    // );
    println!("Dataset loaded");
    let bitset = RSparseBitsetStructure::format_input_data(&dataset);
    let mut structure = RSparseBitsetStructure::new(&bitset, 0);
    structure.push((20, 1));
    println!("Value : {:?}", structure.labels_support());
    structure.backtrack();
    println!("Value : {:?}", structure.parallel_temp_push((20, 1)));
    println!("Value : {:?}", structure.parallel_temp_push((20, 1)));

    // let candidates = (0usize..structure.num_attributes()).collect_vec();
    // let start = Instant::now();
    // let  c = MurTree::build_depth_two_matrix(&mut structure, &candidates, );
    // println!("Time: {:?}", start.elapsed().as_millis());
    // println!("{:?}", c.len());
    // let dataset = BinaryDataset::load(
    //     "experiments/data/parallel_datasets/250_1000000.csv",
    //     false,
    //     0.0,
    // );
    //
    // let n_threads = 6;
    // //let sizes = [1000usize, 2000, 5000, 10_000, 100_000, 200_000, 500_000, 700_000, 800_000, 1_000_000, 2_000_000, 3_000_000, 7_000_000, 10_000_0000];
    // //let size = 7_000;
    // let sizes = [100usize, 1000];
    // let repeat = 100;
    //
    // let mut results = vec![];
    //
    // for size in sizes.iter(){
    //     let a = gen_random_vec(*size);
    //     let b = gen_random_vec(*size);
    //     let mut thread_res = vec![];
    //     for thread in 1..n_threads{
    //         let mut total = 0f64;
    //         for count in 0..repeat {
    //             let start = Instant::now();
    //             let c = compare(&a, &b, thread);
    //             let end = Instant::now();
    //             let elapsed = end.duration_since(start).as_secs_f64() * 1000.0;
    //             total += elapsed;
    //         }
    //         thread_res.push(total/(repeat as f64));
    //
    //         println!("Mean duration for {} threads is {:?} ms", thread, total/(repeat as f64));
    //     }
    //     results.push(thread_res);
    // }
    //
    // let res = ExpeRes{size: sizes.to_vec(), res: results};
    // res.to_json("experiments/data/output_pop_count_parallel.json".to_string());
    //

    // let start = Instant::now();
    // let n = 20_000;
    // let tab = (0..n).collect::<Vec<usize>>();
    // let duration = start.elapsed().as_millis();

    // Using rayon to compute the sum in parallel with min length of 1000
    // let start = Instant::now();
    // // let sum = tab.par_iter().sum::<usize>();
    // let sum = tab.par_chunks(n / n_threads).map(|chunk| chunk.iter().sum::<usize>()).sum::<usize>();
    // let duration = start.elapsed().as_millis();
    // println!("Sum: {:?}", sum);
    // println!("Duration for para : {:?}", duration);

    // let mut handles = vec![];
    // let mut value = 0;
    //
    // let n_repeat = 10;
    // for n_thread in 2..9 {
    //     let mut total_duration = 0f64;
    //     let chunk_size = tab.len() / n_thread;
    //
    //     let start = Instant::now();
    //     for _ in 0..n_repeat {
    //
    //         let _ = thread::scope(|s| {
    //             let mut handles = vec![];
    //             for i in 0..n_thread {
    //
    //                 let chunk_start = i * chunk_size;
    //                 let chunk_end = if i == n_thread - 1 {
    //                     tab.len()
    //                 } else {
    //                     (i + 1) * chunk_size
    //                 };
    //
    //                 let chunk = &tab[chunk_start..chunk_end];
    //                 handles.push(s.spawn( move || {
    //                     chunk.iter().sum::<usize>()
    //                 }));
    //             }
    //             value = handles.into_iter().map(|handle| handle.join().unwrap()).sum::<usize>();
    //
    //         });
    //     }
    //     total_duration += start.elapsed().as_millis() as f64 ;
    //     println!("Duration for para with {} threads: {:?} ms", n_thread, total_duration / n_repeat as f64);
    //     println!("Sum: {:?}", value);
    //     value = 0;
    // }

    // let start = Instant::now();
    // for i in 0..num_threads {
    //     let chunk_start = i * chunk_size;
    //     let chunk_end = if i == num_threads - 1 {
    //         tab.len()
    //     } else {
    //         (i + 1) * chunk_size
    //     };
    //
    //     let chunk = &tab[chunk_start..chunk_end];
    //
    //     let handle = thread::spawn(|| chunk.iter().sum::<usize>());
    //     handles.push(handle);
    // }

    // Collect the results from each thread and compute the final sum
    // let sum: usize = handles.into_iter().map(|handle| handle.join().unwrap()).sum();
    // let duration = start.elapsed().as_millis();
    // println!("Sum: {:?}", arc_val.lock().unwrap());
    // println!("Duration for para : {:?}", duration);

    // Sequential sum

    // Parallelize the sum using a number of threads

    // println!("Tab size: {:?}", tab.len());
    // let start = Instant::now();
    // let sum = tab.iter().sum::<usize>();
    // let duration = start.elapsed().as_millis();
    // println!("Sum: {:?}", sum);
    // println!("Duration sequential: {:?}", duration);
    // //
    // let bitset_data = RSparseBitsetStructure::format_input_data(&dataset);
    // let mut structure = RSparseBitsetStructure::new(&bitset_data);
    // let num_attributes = structure.num_attributes();
    // println!("Num attributes: {:?}", num_attributes);
    // println!("Num labels: {:?}", structure.num_labels());
    // println!("Support: {:?}", structure.support());
    // let n = 100;
    // let mut total_duration = 0f64;
    //
    // for n_thread in 2..15 {
    //     total_duration = 0f64;
    //     let mut total_count = 0;
    //     for _ in 0..n {
    //         let start = Instant::now();
    //         total_count += structure.parallel_temp_push((20, 1), n_thread);
    //         total_duration += start.elapsed().as_micros() as f64;
    //     }
    //     println!("Parallel temp push time with {} threads: {:?} us with count = {}", n_thread, total_duration / n as f64, total_count);
    //
    // }
    // println!();
    // println!();
    // for n_thread in 2..15 {
    //     total_duration = 0f64;
    //     let mut total_count = 0;
    //     for _ in 0..n {
    //         let start = Instant::now();
    //         total_count += structure.parallel_temp_push_v2((20, 1), n_thread);
    //         total_duration += start.elapsed().as_micros() as f64;
    //     }
    //     println!("Parallel temp push time v2 with {} threads: {:?} us and tt = {}", n_thread, total_duration / n as f64, total_count);
    //
    // }
    //
    // println!();
    // println!();
    //
    // total_duration = 0f64;
    // let mut total_count = 0;
    // for _ in 0..n {
    //     let start = Instant::now();
    //     total_count += structure.temp_push((20, 1));
    //     total_duration += start.elapsed().as_micros() as f64;
    // }
    // println!("Sequential temp push time: {:?} us with total count = {}", total_duration / n as f64, total_count);

    // let mut heuristic: Box<dyn Heuristic> = Box::new(NoHeuristic::default());

    // let mut algo: DL85<'_, _, Data> = DL85::new(
    //     1,
    //     2,
    //     <usize>::MAX,
    //     600,
    //     Specialization::None,
    //     LowerBoundHeuristic::None,
    //     BranchingType::None,
    //     CacheInit::WithMemoryDynamic,
    //     0,
    //     true,
    //     heuristic.as_mut(),
    // );

    // let algo = LGDT::fit(&mut structure, 5, 2, InfoGain::fit);
    // algo.print();
    // algo.fit(&mut structure);
    // algo.tree.print();
}
