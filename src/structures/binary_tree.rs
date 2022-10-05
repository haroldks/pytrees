use crate::structures::structures_types::{Attribute, Depth, TreeIndex};
use rand::distributions::Slice;

#[derive(Debug)]
pub struct NodeData<V> {
    pub(crate) test: Attribute,
    pub(crate) metric: V,
    pub(crate) out: Option<usize>,
}

pub struct TreeNode<T> {
    pub value: T,
    pub(crate) index: TreeIndex,
    pub(crate) left: usize,
    pub(crate) right: usize,
}

pub struct Tree<T> {
    tree: Vec<TreeNode<T>>,
}

impl<T> Default for Tree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Tree<T> {
    // TODO: functions for fixed size tree

    pub fn new() -> Self {
        Tree { tree: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    pub(crate) fn add_node(
        &mut self,
        parent: TreeIndex,
        is_left: bool,
        mut node: TreeNode<T>,
    ) -> TreeIndex {
        node.index = self.tree.len();
        self.tree.push(node);
        let position = self.tree.len() - 1;
        if position == 0 {
            return position;
        }
        if let Some(parent_node) = self.tree.get_mut(parent) {
            if is_left {
                parent_node.left = position
            } else {
                parent_node.right = position
            }
        };
        position
    }

    pub fn add_root(&mut self, root: TreeNode<T>) -> TreeIndex {
        self.add_node(0, false, root)
    }

    pub fn add_left_node(&mut self, parent: TreeIndex, node: TreeNode<T>) -> TreeIndex {
        self.add_node(parent, true, node)
    }
    pub fn add_right_node(&mut self, parent: TreeIndex, node: TreeNode<T>) -> TreeIndex {
        self.add_node(parent, false, node)
    }

    pub fn get_root_index(&self) -> TreeIndex {
        0
    }

    pub fn get_node(&self, index: TreeIndex) -> Option<&TreeNode<T>> {
        self.tree.get(index)
    }

    pub fn get_node_mut(&mut self, index: TreeIndex) -> Option<&mut TreeNode<T>> {
        self.tree.get_mut(index)
    }

    pub fn get_left_child(&self, node: &TreeNode<T>) -> Option<&TreeNode<T>> {
        if node.left == 0 {
            None
        } else {
            self.tree.get(node.left)
        }
    }
    pub fn get_left_child_mut(&mut self, node: &TreeNode<T>) -> Option<&mut TreeNode<T>> {
        // FIXME: Might cause issues later
        if node.left == 0 {
            None
        } else {
            self.tree.get_mut(node.left)
        }
    }

    pub fn get_right_child(&self, node: &TreeNode<T>) -> Option<&TreeNode<T>> {
        if node.right == 0 {
            None
        } else {
            self.tree.get(node.right)
        }
    }
    pub fn get_right_child_mut(&mut self, node: &TreeNode<T>) -> Option<&mut TreeNode<T>> {
        // FIXME: Might cause issues later
        if node.right == 0 {
            None
        } else {
            self.tree.get_mut(node.right)
        }
    }
    pub fn print(&self)
    where
        T: std::fmt::Debug,
    {
        let mut stack: Vec<(usize, Option<&TreeNode<T>>)> = Vec::new();
        let root = self.get_node(self.get_root_index());
        stack.push((0, root));
        while !stack.is_empty() {
            let next = stack.pop();
            if let Some((deep, node_opt)) = next {
                if let Some(node) = node_opt {
                    for _i in 0..deep {
                        print!("    ");
                    }
                    println!("----{:?}", node.value);

                    stack.push((deep + 1, self.get_right_child(node)));
                    stack.push((deep + 1, self.get_left_child(node)));
                }
            }
        }
    }
}
