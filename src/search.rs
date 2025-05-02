use rand::seq::IndexedRandom;

#[derive(PartialEq, Eq, Hash)]
pub struct HashKey<const N: usize>([u32; N]);

#[derive(Clone, Copy)]
pub struct Vector<const N: usize>([f32; N]);

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
        indexes: &Vec<usize>,
        all_vecs: &Vec<Vector<N>>,
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
}
