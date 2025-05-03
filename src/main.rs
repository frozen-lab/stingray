use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use stingray::search::{ANNIndex, Vector};

fn load_embeddings(path: &str) -> HashMap<String, Vec<f32>> {
    let file = File::open(path).expect("cannot open embeddings");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("invalid JSON format")
}

fn main() {
    const N: usize = 50;

    let map = load_embeddings("assets/vec.json");

    let mut vecs: Vec<Vector<N>> = Vec::with_capacity(map.len());
    let mut ids: Vec<i32> = Vec::with_capacity(map.len());
    let mut words: Vec<String> = Vec::with_capacity(map.len());

    for (i, (word, coords)) in map.into_iter().enumerate() {
        // sanity check
        assert_eq!(coords.len(), N, "vector size mismatch for `{}`", word);

        let arr: [f32; N] = coords.try_into().expect("slice with incorrect length");
        vecs.push(Vector(arr));
        ids.push(i as i32);
        words.push(word);
    }

    let ann: ANNIndex<N> = ANNIndex::<N>::build_index(10, 30, &vecs, &ids);

    let query: Vector<N> = vecs[0];

    let top_k = 5;

    let results: Vec<(i32, f32)> = ann.search_approximate(query, top_k);

    println!("Nearest neighbors of `{}`:", words[0]);

    for (id, dist) in results {
        println!("  {:<15}  (distance = {:.4})", words[id as usize], dist);
    }
}
