pub type Support = usize;
pub type Depth = usize;
pub type Attribute = usize;
pub type Item = (Attribute, usize);
pub type Position = Vec<Item>;
pub static MAX_INT: usize = <usize>::MAX;

// Horizontal data structure type
pub type HorizontalData = Vec<Vec<Vec<usize>>>;
pub type HBSStackState = Vec<Vec<Vec<usize>>>;
pub type HBSState = Vec<Vec<usize>>; // A stack containing the vectors used for counting and so one

// Bitsets data structure type
pub type Bitset = Vec<u64>;
pub type BitsetMatrix = Vec<Bitset>;

pub struct BitsetStructData {
    pub(crate) inputs: BitsetMatrix,
    pub(crate) targets: BitsetMatrix,
    pub(crate) chunks: usize,
    pub(crate) size: usize,
}

pub type BitsetStackState = Vec<Bitset>;

// Tree types
pub type Index = usize;
pub type StateCollection = Vec<(Vec<Item>, Bitset, Index)>;
// Double Pointer Structure

pub struct DoublePointerData {
    pub(crate) inputs: Vec<Vec<usize>>,
    pub(crate) target: Vec<usize>,
    pub(crate) num_labels: usize,
    pub(crate) num_attributes: usize,
}

#[derive(Debug)]
pub struct LeafInfo {
    pub(crate) index: Index,
    pub(crate) position: Position,
    pub(crate) bitset: Bitset,
    pub(crate) error: usize,
}
