pub type Support = usize;
pub type Attribute = usize;
pub type Item = (Attribute, usize);
pub type Position = Vec<Item>;

// Horizontal data structure type
pub type HorizontalData = Vec<Vec<Vec<usize>>>;
pub type HBSStackState = Vec<Vec<Vec<usize>>>;
pub type HBSState = Vec<Vec<usize>>; // A stack containing the vectors used for counting and so one
