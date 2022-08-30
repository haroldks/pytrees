use crate::structures::structure_trait::Structure;
use crate::structures::structures_types::{Attribute, Depth, Support};

trait Algorithm {
    fn fit<S>(structure: &mut S, support: Support, depth: Depth)
    where
        S: Structure;

    fn first_candidates<S>(structure: &mut S, min_sup: Support) -> Vec<Attribute>
    where
        S: Structure,
    {
        let num_attributes = structure.num_attributes();
        let mut candidates = vec![];
        for i in 0..num_attributes {
            if structure.temp_push((i, 0)) >= min_sup && structure.temp_push((i, 1)) >= min_sup {
                candidates.push(i);
            }
        }
        candidates
    }

    fn sort_candidates<S, F>(
        structure: &mut S,
        candidates: &Vec<Attribute>,
        func: F,
        increasing: bool,
    ) -> Vec<Attribute>
    where
        S: Structure,
        F: Fn(&mut S, &Vec<Attribute>, bool) -> Vec<Attribute>,
    {
        func(structure, candidates, increasing)
    }
}
