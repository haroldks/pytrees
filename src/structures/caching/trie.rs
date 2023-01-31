use crate::structures::structures_types::{Attribute, Depth, Index, Item, MAX_INT};
use nohash_hasher::BuildNoHashHasher;
use std::collections::{BTreeSet, HashMap};

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
}

#[derive(Copy, Clone, Debug)]
pub struct Data {
    // TODO: Should use float ?
    pub test: Attribute,
    pub depth: Depth,
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
}

#[derive(Copy, Clone, Debug)]
pub struct TrieNode<T> {
    pub item: Item,
    pub value: T,
    pub index: Index,
}

impl<T> TrieNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            item: (MAX_INT, 0),
            value,
            index: 0,
        }
    }
}

// Begin: Iterator On Node Children

struct ChildrenIter<'a, It> {
    cache: &'a Trie<It>,
    to_explore: &'a Vec<usize>,
    count: usize,
    limit: usize,
}

impl<'a, It> ChildrenIter<'a, It> {
    pub fn new(cache: &'a Trie<It>, to_explore: &'a Vec<usize>) -> Self {
        Self {
            cache,
            to_explore,
            count: 0,
            limit: to_explore.len(),
        }
    }
}

impl<'a, It> Iterator for ChildrenIter<'a, It> {
    type Item = &'a TrieNode<It>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.limit {
            let index = self.to_explore[self.count];
            self.count += 1;
            return Some(&self.cache.cache[index]);
        }
        None
    }
}

// End: Iterator On Node Children
#[derive(Debug)]
pub struct Trie<T> {
    cache: Vec<TrieNode<T>>,
    children: HashMap<usize, Vec<usize>, BuildNoHashHasher<usize>>,
}

impl<T: DataTrait> Default for Trie<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataTrait> Trie<T> {
    pub fn new() -> Self {
        Self {
            cache: Vec::new(),
            children: Default::default(),
        }
    }

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
    fn children(&self, index: Index) -> Option<ChildrenIter<T>> {
        let node_children = self.children.get(&index);
        node_children.map(|children| ChildrenIter::new(self, children))
    }

    fn add_child(&mut self, parent: Index, child_index: Index) {
        self.children
            .entry(parent)
            .and_modify(|node_children| node_children.push(child_index))
            .or_insert_with(|| vec![child_index]);
    }

    // Start: Cache Exploration based on Itemset
    pub fn find<'a, I: Iterator<Item = &'a (usize, usize)>>(&self, itemset: I) -> Option<Index> {
        let mut index = self.get_root_index();
        for item in itemset {
            let children = self.children(index);
            match children {
                None => {
                    return None;
                }
                Some(iterator) => {
                    let mut found = false;
                    for child in iterator {
                        if child.item == *item {
                            index = child.index;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return None;
                    }
                }
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
            match children {
                None => {
                    new = true;
                    index = self.create_cache_entry(index, item);
                }
                Some(iterator) => {
                    let mut found = false;
                    for child in iterator {
                        if child.item == *item {
                            index = child.index;
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

#[cfg(test)]
mod trie_test {
    use crate::structures::caching::trie::{Data, DataTrait, Trie, TrieNode};

    #[test]
    fn test_check_children_index_iterator() {
        let mut cache: Trie<Data> = Trie::new();
        let value = Data::new();
        let node = TrieNode::new(value);

        cache.add_root(node);

        let value = Data::new();
        let node = TrieNode::new(value);

        cache.add_node(0, node);
        cache.add_node(0, node);

        let child_iterator = cache.children(0);
        assert_eq!(child_iterator.is_some(), true);
        if let Some(list) = child_iterator {
            assert_eq!(list.limit, 2);
        }
        let a = [(0, 0), (1, 1), (2, 0)].iter();
        let index = cache.find_or_create(a);

        let data = Data {
            test: 0,
            depth: 0,
            error: 0,
            error_as_leaf: 0,
            lower_bound: 0,
            out: 33,
            is_leaf: false,
            metric: None,
        };
        let a = [(0, 0), (1, 1), (2, 0)].iter();
        cache.update(a, data);
    }
}
