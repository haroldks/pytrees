pub type Support = usize;
pub type Depth = usize;
pub type Attribute = usize;
pub type Item = (Attribute, usize);
pub type Position = Vec<Item>;

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
pub type TreeIndex = usize;

// Double Pointer Structure

pub struct DoublePointerData {
    pub(crate) inputs: Vec<Vec<usize>>,
    pub(crate) target: Vec<usize>,
    pub(crate) num_labels: usize,
    pub(crate) num_attributes: usize,
}
