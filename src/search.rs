use dashmap::DashSet;
use itertools::Itertools;
use rand::seq::IndexedRandom;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{cmp::min, collections::HashSet};

#[derive(PartialEq, Eq, Hash)]
pub struct HashKey<const N: usize>([u32; N]);

#[derive(Clone, Copy)]
pub struct Vector<const N: usize>(pub [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped = self.0.iter().zip(vector.0).map(|(a, b)| b - a);
        let coords = mapped.collect::<Vec<_>>().try_into().unwrap();

        Vector(coords)
    }

    pub fn avg(&self, vecotr: &Vector<N>) -> Vector<N> {
        let mapped = self.0.iter().zip(vecotr.0).map(|(a, b)| (a + b) / 2.0);
        let cords = mapped.collect::<Vec<_>>().try_into().unwrap();

        Vector(cords)
    }

    pub fn dot_product(&self, vector: &Vector<N>) -> f32 {
        self.0.iter().zip(vector.0).map(|(a, b)| a * b).sum::<f32>()
    }

    pub fn to_hashkey(&self) -> HashKey<N> {
        let mapped = self.0.iter().map(|f| f.to_bits());
        let coords: [u32; N] = mapped.collect::<Vec<_>>().try_into().unwrap();

        HashKey::<N>(coords)
    }

    pub fn sq_euc_dis(&self, vecotr: &Vector<N>) -> f32 {
        self.0
            .iter()
            .zip(vecotr.0)
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }
}

struct Hyperplane<const N: usize> {
    coefficient: Vector<N>,
    constant: f32,
}

impl<const N: usize> Hyperplane<N> {
    fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficient.dot_product(point) + self.constant >= 0.0
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}

struct InnerNode<const N: usize> {
    hyperplane: Hyperplane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}

struct LeafNode<const N: usize>(Vec<usize>);

pub struct ANNIndex<const N: usize> {
    ids: Vec<i32>,
    trees: Vec<Node<N>>,
    values: Vec<Vector<N>>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &[usize],
        all_vecs: &[Vector<N>],
    ) -> (Hyperplane<N>, Vec<usize>, Vec<usize>) {
        let sample: Vec<_> = indexes.choose_multiple(&mut rand::rng(), 2).collect();

        let (a, b) = (*sample[0], *sample[1]);
        let coefficient = all_vecs[a].subtract_from(&all_vecs[b]);
        let midpoint = all_vecs[a].avg(&all_vecs[b]);
        let constant = -coefficient.dot_product(&midpoint);

        let hyperplane = Hyperplane {
            coefficient,
            constant,
        };

        let (mut below, mut above) = (vec![], vec![]);

        for &id in indexes.iter() {
            if hyperplane.point_is_above(&all_vecs[id]) {
                above.push(id);
            } else {
                below.push(id);
            }
        }

        (hyperplane, above, below)
    }

    fn build_tree(max_size: usize, indexes: &[usize], all_vecs: &Vec<Vector<N>>) -> Node<N> {
        if indexes.len() <= max_size {
            return Node::Leaf(Box::new(LeafNode::<N>(indexes.to_owned())));
        }

        let (hyperplane, above, below) = Self::build_hyperplane(indexes, all_vecs);

        let left_node = Self::build_tree(max_size, &above, all_vecs);
        let right_node = Self::build_tree(max_size, &below, all_vecs);

        Node::Inner(Box::new(InnerNode::<N> {
            hyperplane,
            left_node,
            right_node,
        }))
    }

    fn deduplicate(
        vecs: &[Vector<N>],
        ids: &[i32],
        dedup_vecs: &mut Vec<Vector<N>>,
        dedup_ids: &mut Vec<i32>,
    ) {
        let mut seen_ids: HashSet<HashKey<N>> = HashSet::new();

        for i in 1..vecs.len() {
            let hash_key = vecs[i].to_hashkey();

            if !seen_ids.contains(&hash_key) {
                seen_ids.insert(hash_key);
                dedup_vecs.push(vecs[i]);
                dedup_ids.push(ids[i]);
            }
        }
    }

    fn tree_result(query: Vector<N>, n: i32, tree: &Node<N>, candidates: &DashSet<usize>) -> i32 {
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(box_leaf.0);
                let num_candidates_found = min(n as usize, leaf_values.len());

                for item in leaf_values.iter().take(num_candidates_found) {
                    candidates.insert(*item);
                }

                num_candidates_found as i32
            }
            Node::Inner(inner) => {
                let above = (inner).hyperplane.point_is_above(&query);
                let (main, backup) = match above {
                    true => (&(inner.right_node), &(inner.left_node)),
                    false => (&(inner.left_node), &(inner.right_node)),
                };

                match Self::tree_result(query, n, main, candidates) {
                    k if k < n => k + Self::tree_result(query, n - k, backup, candidates),
                    k => k,
                }
            }
        }
    }

    pub fn build_index(
        num_trees: usize,
        max_size: usize,
        vecs: &[Vector<N>],
        ids: &[i32],
    ) -> ANNIndex<N> {
        let (mut unique_vecs, mut unique_ids) = (vec![], vec![]);
        Self::deduplicate(vecs, ids, &mut unique_vecs, &mut unique_ids);

        let all_indexes: Vec<usize> = (0..unique_vecs.len()).collect();
        let trees: Vec<_> = (0..num_trees)
            .into_par_iter()
            .map(|_| Self::build_tree(max_size, &all_indexes, &unique_vecs))
            .collect();

        ANNIndex::<N> {
            values: unique_vecs,
            trees,
            ids: unique_ids,
        }
    }

    pub fn search_approximate(&self, query: Vector<N>, top_k: i32) -> Vec<(i32, f32)> {
        let candidates = DashSet::new();

        self.trees.par_iter().for_each(|tree| {
            Self::tree_result(query, top_k, tree, &candidates);
        });

        candidates
            .into_iter()
            .map(|idx| (idx, self.values[idx].sq_euc_dis(&query)))
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .take(top_k as usize)
            .map(|(idx, dis)| (self.ids[idx], dis))
            .collect()
    }
}
