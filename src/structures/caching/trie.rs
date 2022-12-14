use crate::structures::binary_tree::Tree;
use crate::structures::structures_types::{Attribute, CacheIndex, Depth, Item, MAX_INT};
use ndarray::s;
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use std::slice::Iter;

trait DataTrait {
    fn new() -> Self;
    fn create_on_item(item: &Item) -> Self;
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
}

#[derive(Copy, Clone, Debug)]
struct TrieNode<T> {
    pub item: Item,
    pub value: T,
    pub index: CacheIndex,
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

// TODO: Maybe useless
// impl <'a, It,> IntoIterator for ChildrenIter<'a, It> {
//     type Item = &'a TrieNode<It>;
//     type IntoIter = ChildrenIter<'a, It>;
//
//     fn into_iter(self) -> ChildrenIter<'a, It> {
//       ChildrenIter {
//           cache: self.cache,
//           to_explore: self.to_explore,
//           count: self.count,
//           limit: self.limit,
//       }
//     }
// }
// TODO: End Maybe useless

// End: Iterator On Node Children

struct Trie<T> {
    cache: Vec<TrieNode<T>>,
    children: HashMap<usize, Vec<usize>, BuildNoHashHasher<usize>>,
}

// impl<T: Da> Default for Trie<T> {
//     fn default() -> Self {
//         Self::new()
//     }
// }

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

    pub fn add_node(&mut self, parent: CacheIndex, mut node: TrieNode<T>) -> CacheIndex {
        node.index = self.cache.len();
        self.cache.push(node);
        let position = self.cache.len() - 1;
        if position == 0 {
            return position;
        }
        self.add_child(parent, position);
        position
    }

    pub fn add_root(&mut self, root: TrieNode<T>) -> CacheIndex {
        self.add_node(0, root)
    }

    pub fn get_root_index(&self) -> CacheIndex {
        0
    }

    pub fn get_node(&self, index: CacheIndex) -> Option<&TrieNode<T>> {
        self.cache.get(index)
    }

    pub fn get_node_mut(&mut self, index: CacheIndex) -> Option<&mut TrieNode<T>> {
        self.cache.get_mut(index)
    }

    // End : Index based methods

    // NodeIndex : Get Iterator
    fn children(&self, index: CacheIndex) -> Option<ChildrenIter<T>> {
        let node_children = self.children.get(&index);
        let iterator = match node_children {
            None => None,
            Some(children) => Some(ChildrenIter::new(self, children)),
        };
        iterator
    }

    fn has_children(&self, node_index: CacheIndex) -> bool {
        // TODO: Useless ?
        if self.children.contains_key(&node_index) {
            if let Some(node_children) = self.children.get(&node_index) {
                return node_children.len() > 0;
            }
        }
        false
    }

    fn add_child(&mut self, parent: CacheIndex, child_index: CacheIndex) {
        self.children
            .entry(parent)
            .and_modify(|node_children| node_children.push(child_index))
            .or_insert(vec![child_index]);
    }

    // Start: Cache Exploration based on Itemset

    pub fn find(&self, itemset: &[Item]) -> Option<CacheIndex> {
        let mut index = self.get_root_index();
        for item in itemset.iter() {
            let mut children = self.children(index);
            match children {
                None => {
                    return None;
                }
                Some(iterator) => {
                    let mut found = false;
                    for child in iterator.into_iter() {
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

    pub fn find_or_create(&mut self, itemset: &[Item]) -> CacheIndex {
        let mut index = self.get_root_index();
        for item in itemset.iter() {
            let mut children = self.children(index);
            match children {
                None => {
                    index = self.create_cache_entry(index, item);
                }
                Some(iterator) => {
                    let mut found = false;
                    for child in iterator.into_iter() {
                        if child.item == *item {
                            index = child.index;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        // TODO : Check if possible to not do this
                        index = self.create_cache_entry(index, item);
                    }
                }
            }
        }
        index
    }

    fn create_cache_entry(&mut self, parent: CacheIndex, item: &Item) -> CacheIndex {
        let mut data = T::create_on_item(item);
        let mut node = TrieNode::new(data);
        node.item = *item;
        self.add_node(parent, node)
    }

    pub fn update(&mut self, itemset: &[Item], data: T) {
        let index = self.find(itemset);
        if let Some(node_index) = index {
            if let Some(node) = self.get_node_mut(node_index) {
                node.value = data;
            }
        }
    }

    // Start: Cache Exploration based on Itemset
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
        let index = cache.find_or_create(&[(0, 0), (1, 1), (2, 0)]);
        // println!("{:?}", index);

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

        cache.update(&[(0, 0), (1, 1), (2, 0)], data);

        //
        // let index = cache.find(&[(0, 0),(1, 1)]);
        // println!("{:?}", index);
        // let index = cache.find(&[(1, 1)]);
        // println!("{:?}", index);
        // println!("{:?}", cache.get_node(5));
    }
}
