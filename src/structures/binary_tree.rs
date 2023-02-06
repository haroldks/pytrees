use crate::structures::structures_types::{Attribute, Index};
use pyo3::{IntoPy, PyObject, Python};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NodeData {
    // Specific data for decision trees
    pub(crate) test: Option<Attribute>,
    pub(crate) error: usize,
    pub(crate) metric: Option<f64>,
    pub(crate) out: Option<usize>,
}

impl Default for NodeData {
    fn default() -> Self {
        NodeData::new()
    }
}

impl NodeData {
    pub fn new() -> NodeData {
        NodeData {
            test: None,
            error: <usize>::MAX,
            metric: None,
            out: None,
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub struct TreeNode<T> {
    pub value: T,
    pub(crate) index: Index,
    pub(crate) left: usize,
    pub(crate) right: usize,
}

impl<T> TreeNode<T> {
    pub fn new(value: T) -> TreeNode<T> {
        TreeNode {
            value,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Tree<T> {
    tree: Vec<TreeNode<T>>,
}

impl<T> Default for Tree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoPy<PyObject> for Tree<NodeData> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let json = serde_json::to_string_pretty(&self).unwrap();
        json.into_py(py)
    }
}

impl<T> Tree<T> {
    pub fn new() -> Self {
        Tree { tree: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Tree {
            tree: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    pub fn actual_len(&self) -> usize {
        self.count_node_recursion(self.get_root_index())
    }

    fn count_node_recursion(&self, node_index: Index) -> usize {
        let mut left_index = 0;
        let mut right_index = 0;
        if let Some(node) = self.get_node(node_index) {
            if node.left == node.right {
                return 1;
            } else {
                left_index = node.left;
                right_index = node.right;
            }
        }

        let mut count = 0;

        if left_index != 0 {
            count += self.count_node_recursion(left_index);
        }
        if right_index != 0 {
            count += self.count_node_recursion(right_index);
        }

        count + 1
    }

    pub(crate) fn add_node(
        &mut self,
        parent: Index,
        is_left: bool,
        mut node: TreeNode<T>,
    ) -> Index {
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

    pub fn add_root(&mut self, root: TreeNode<T>) -> Index {
        self.add_node(0, false, root)
    }

    pub fn add_left_node(&mut self, parent: Index, node: TreeNode<T>) -> Index {
        self.add_node(parent, true, node)
    }
    pub fn add_right_node(&mut self, parent: Index, node: TreeNode<T>) -> Index {
        self.add_node(parent, false, node)
    }

    pub fn get_root_index(&self) -> Index {
        0
    }

    pub fn get_node(&self, index: Index) -> Option<&TreeNode<T>> {
        self.tree.get(index)
    }

    pub fn get_node_mut(&mut self, index: Index) -> Option<&mut TreeNode<T>> {
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

#[cfg(test)]
mod binary_tree_test {
    use crate::structures::binary_tree::{NodeData, Tree, TreeNode};

    #[test]
    fn create_node_data() {
        let data = NodeData::new();
        assert_eq!(data.error, <usize>::MAX);
        assert_eq!(data.test.is_none(), true);
        assert_eq!(data.out, None);
    }

    #[test]
    fn create_tree_node() {
        let data = NodeData::new();
        let node = TreeNode::new(data);
        assert_eq!(node.right, 0);
        assert_eq!(node.left, 0);
        assert_eq!(node.index, 0);
    }

    #[test]
    fn tree_default() {
        let tree: Tree<i32> = Tree::default();
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn tree_new() {
        let tree: Tree<i32> = Tree::default();
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn tree_is_empty() {
        let mut tree: Tree<i32> = Tree::new();
        assert_eq!(tree.is_empty(), true);

        let root = TreeNode::new(5);
        tree.add_root(root);
        assert_eq!(tree.is_empty(), false);
    }

    #[test]
    fn binarytree_add_root() {
        let mut tree: Tree<f32> = Tree::new();
        let root = TreeNode::new(10.0);
        let root_index = tree.add_root(root);
        assert_eq!(0, root_index);
    }

    #[test]
    fn binarytree_get_root_index() {
        let mut tree: Tree<f32> = Tree::new();
        let root = TreeNode::new(10.0);
        let _ = tree.add_root(root);
        let root_index = tree.get_root_index();
        assert_eq!(0, root_index);
    }

    #[test]
    fn binarytree_get_left_child() {
        let mut tree: Tree<f32> = Tree::new();
        let root = TreeNode::new(10.0);
        let root_index = tree.add_root(root);
        let left_node = TreeNode::new(5.0);
        let _ = tree.add_left_node(root_index, left_node);
        let root = tree.get_node(root_index).unwrap();
        let left_node = tree.get_left_child(root).unwrap();
        assert_eq!(left_node.value, 5.0);
    }

    #[test]
    fn binarytree_get_right_child() {
        let mut tree: Tree<f32> = Tree::new();
        let root = TreeNode::new(10.0);
        let root_index = tree.add_root(root);
        let right_node = TreeNode::new(5.0);
        let _ = tree.add_right_node(root_index, right_node);
        let root = tree.get_node(root_index).unwrap();
        let right_node = tree.get_right_child(root).unwrap();
        assert_eq!(right_node.value, 5.0);
    }

    #[test]
    fn test_get_node() {
        let mut tree: Tree<i32> = Tree::new();
        let root = TreeNode::new(10);
        let _ = tree.add_root(root);
        let root_index = tree.get_root_index();
        let root = tree.get_node(root_index).unwrap();
        assert_eq!(10, root.value)
    }

    #[test]
    fn test_get_node_mut() {
        let mut tree: Tree<i32> = Tree::new();
        let root = TreeNode::new(10);
        let _ = tree.add_root(root);
        let root_index = tree.get_root_index();
        let root = tree.get_node_mut(root_index).unwrap();
        root.value = 11;
        assert_eq!(11, root.value);
    }

    #[test]
    fn test_add_left_node() {
        let mut tree: Tree<f32> = Tree::new();
        let root = TreeNode::new(10.0);
        let root_index = tree.add_root(root);
        let left_node = TreeNode::new(5.0);
        let _ = tree.add_left_node(root_index, left_node);
        let root = tree.get_node(root_index).unwrap();
        let left_node = tree.get_left_child(root).unwrap();
        assert_eq!(left_node.value, 5.0);
    }

    #[test]
    fn test_add_right_node() {
        let mut tree: Tree<f32> = Tree::new();
        let root = TreeNode::new(10.0);
        let root_index = tree.add_root(root);
        let right_node = TreeNode::new(5.0);
        let _ = tree.add_right_node(root_index, right_node);
        let root = tree.get_node(root_index).unwrap();
        let right_node = tree.get_right_child(root).unwrap();
        assert_eq!(right_node.value, 5.0);
    }
}
