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

