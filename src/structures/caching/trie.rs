use crate::structures::structures_types::{Attribute, Depth, Index, Item, MAX_INT};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{BTreeSet, HashMap};
use std::slice::Iter;

static MEMORY_SIZE: usize = 2_000_000_000; // 2GB

pub trait DataTrait {
    fn new() -> Self;
    fn create_on_item(item: &Item) -> Self;
    fn get_node_error(&self) -> usize;
    fn get_leaf_error(&self) -> usize;
    fn set_node_error(&mut self, error: usize);
    fn set_leaf_error(&mut self, error: usize);
    fn set_test(&mut self, test: Attribute);
    fn set_class(&mut self, class: usize);
    fn get_class(&self) -> usize;
    fn get_lower_bound(&self) -> usize;
    fn set_lower_bound(&mut self, lower_bound: usize);
    fn get_test(&self) -> Attribute;
    fn to_leaf(&mut self);
    fn is_leaf(&self) -> bool;
    fn set_as_leaf(&mut self);
    fn set_discrepancy(&mut self, discrepancy: usize);
    fn get_discrepancy(&self) -> usize;
}

#[derive(Copy, Clone, Debug)]
pub struct Data {
    // TODO: Should use float ?
    pub test: Attribute,
    pub depth: Depth,
    pub current_discrepancy: usize,
    pub error: usize,
    pub error_as_leaf: usize,
    pub lower_bound: usize,
    pub out: usize,
    pub is_leaf: bool,
    pub metric: Option<f64>,
}

impl Default for Data {
    fn default() -> Self {
        Self::new()
    }
}

impl DataTrait for Data {
    fn new() -> Self {
        Self {
            test: MAX_INT,
            depth: 0,
            current_discrepancy: 0,
            error: MAX_INT,
            error_as_leaf: MAX_INT,
            lower_bound: 0,
            out: MAX_INT,
            is_leaf: false,
            metric: None,
        }
    }

    fn create_on_item(item: &Item) -> Self {
        let mut data = Self::new();
        data.test = item.0;
        data
    }

    fn get_node_error(&self) -> usize {
        self.error
    }

    fn get_leaf_error(&self) -> usize {
        self.error_as_leaf
    }

    fn set_node_error(&mut self, error: usize) {
        self.error = error;
    }

    fn set_leaf_error(&mut self, error: usize) {
        self.error_as_leaf = error;
    }

    fn set_test(&mut self, test: Attribute) {
        self.test = test;
    }

    fn set_class(&mut self, class: usize) {
        self.out = class;
    }

    fn get_class(&self) -> usize {
        self.out
    }

    fn get_lower_bound(&self) -> usize {
        self.lower_bound
    }

    fn set_lower_bound(&mut self, lower_bound: usize) {
        self.lower_bound = lower_bound;
    }

    fn get_test(&self) -> Attribute {
        self.test
    }

    fn to_leaf(&mut self) {
        self.is_leaf = true;
    }

    fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    fn set_as_leaf(&mut self) {
        self.error = self.error_as_leaf;
        self.is_leaf = true;
    }

    fn set_discrepancy(&mut self, discrepancy: usize) {
        self.current_discrepancy = discrepancy;
    }

    fn get_discrepancy(&self) -> usize {
        self.current_discrepancy
    }
}

#[derive(Clone, Debug)]
pub struct TrieNode<T> {
    // TODO: Add parent index
    pub item: Item,
    pub value: T,
    pub index: Index,
    pub node_children: Vec<Index>,
}

impl<T> TrieNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            item: (MAX_INT, 0),
            value,
            index: 0,
            node_children: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Trie<T> {
    cache: Vec<TrieNode<T>>,
}

impl<T: DataTrait> Default for Trie<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataTrait> Trie<T> {
    pub fn new() -> Self {
        Self {
            cache: Vec::new(), // TODO : Find a better way to set the capacity
        }
    }

    // Start :Implement a better way to set the capacity

    pub fn from_user_memory(size: usize) -> Self {
        if size > MEMORY_SIZE {
            panic!("Memory size is too big! Only 2GB is allowed to start with.");
        }

        let vec_size = size / std::mem::size_of::<TrieNode<T>>();

        Self {
            cache: Vec::with_capacity(vec_size),
        }
    }

    pub fn with_capacity(features: usize, depth: usize) -> Self {
        let cache_size = Self::cache_size(features, depth);

        Self {
            cache: Vec::with_capacity(cache_size),
        }
    }

    fn factorial(n: u64) -> u64 {
        (1..=n).product()
    }

    fn count_combinations(n: u64, r: u64) -> u64 {
        (n - r + 1..=n).product::<u64>() / Self::factorial(r)
    }

    fn cache_size(features: usize, depth: usize) -> usize {
        let mut size = 0;
        for i in 1..depth {
            size += Self::count_combinations(features as u64, i as u64) * 2u64.pow(i as u32);
        }

        let used_bytes = size * std::mem::size_of::<TrieNode<T>>() as u64;

        if used_bytes > MEMORY_SIZE as u64 {
            return MEMORY_SIZE / std::mem::size_of::<TrieNode<T>>();
        }

        size as usize
    }

    // End : Implement a better way to set the capacity

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    fn resize(&mut self) {
        // TODO: Implement Log resize method
        unimplemented!()
    }

    // Begin : Index based methods

    pub fn add_node(&mut self, parent: Index, mut node: TrieNode<T>) -> Index {
        node.index = self.cache.len();
        self.cache.push(node);
        let position = self.cache.len() - 1;
        if position == 0 {
            return position;
        }
        self.add_child(parent, position);
        position
    }

    pub fn add_root(&mut self, root: TrieNode<T>) -> Index {
        self.add_node(0, root)
    }

    pub fn get_root_index(&self) -> Index {
        0
    }

    pub fn get_node(&self, index: Index) -> Option<&TrieNode<T>> {
        self.cache.get(index)
    }

    pub fn get_node_mut(&mut self, index: Index) -> Option<&mut TrieNode<T>> {
        self.cache.get_mut(index)
    }

    // End : Index based methods

    // NodeIndex : Get Iterator
    fn children(&self, index: Index) -> Iter<'_, Index> {
        self.cache[index].node_children.iter()
    }

    fn add_child(&mut self, parent: Index, child_index: Index) {
        self.cache[parent].node_children.push(child_index);
    }

    // Start: Cache Exploration based on Itemset
    pub fn find<'a, I: Iterator<Item = &'a (usize, usize)>>(&self, itemset: I) -> Option<Index> {
        let mut index = self.get_root_index();
        for item in itemset {
            let children = self.children(index);
            let mut found = false;
            for child in children {
                // TODO : Move it to a method
                let node = self.get_node(*child).unwrap();
                if node.item == *item {
                    index = *child;
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
        Some(index)
    }

    pub fn find_or_create<'a, I: Iterator<Item = &'a (usize, usize)>>(
        &mut self,
        itemset: I,
    ) -> (bool, Index) {
        let mut index = self.get_root_index();
        let mut new = false;
        for item in itemset {
            let children = self.children(index);
            let mut found = false;
            for child in children {
                let node = self.get_node(*child).unwrap();
                if node.item == *item {
                    index = *child;
                    found = true;
                    break;
                }
            }
            if !found {
                new = true;
                // TODO : Check if possible to not do this
                index = self.create_cache_entry(index, item);
            }
        }

        (new, index)
    }

    fn create_cache_entry(&mut self, parent: Index, item: &Item) -> Index {
        let data = T::create_on_item(item);
        let mut node = TrieNode::new(data);
        node.item = *item;
        self.add_node(parent, node)
    }

    pub fn update<'a, I: Iterator<Item = &'a (usize, usize)>>(&mut self, itemset: I, data: T) {
        let index = self.find(itemset);
        if let Some(node_index) = index {
            if let Some(node) = self.get_node_mut(node_index) {
                node.value = data;
            }
        }
    }

    // End: Cache Exploration based on Itemset
}
