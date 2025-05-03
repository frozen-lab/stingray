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
    // Assume embeddings all have the same dimension N:
    const N: usize = 50; // set this to your actual vector size

    // after loading:
    let map = load_embeddings("assets/vec.json");

    // reserve space
    let mut vecs: Vec<Vector<N>> = Vec::with_capacity(map.len());
    let mut ids: Vec<i32> = Vec::with_capacity(map.len());
    let mut words: Vec<String> = Vec::with_capacity(map.len());

    for (i, (word, coords)) in map.into_iter().enumerate() {
        // sanity check
        assert_eq!(coords.len(), N, "vector size mismatch for `{}`", word);

        // convert Vec<f32> → [f32; N]
        let arr: [f32; N] = coords.try_into().expect("slice with incorrect length");
        vecs.push(Vector(arr));
        ids.push(i as i32);
        words.push(word);
    }

    // build your index
    let ann: ANNIndex<N> = ANNIndex::<N>::build_index(10, 30, &vecs, &ids);

    // 1. Pick the 0th vector as your query (Vector<N> is Copy)
    let query: Vector<N> = vecs[0];

    // 2. Decide how many neighbors you want
    let top_k = 5;

    // 3. Run your ANN search
    let results: Vec<(i32, f32)> = ann.search_approximate(query, top_k);

    // 4. Print them out, mapping back to words[]
    println!("Nearest neighbors of `{}`:", words[0]);

    for (id, dist) in results {
        // `id` is the original index you stored in ids[], so
        // words[id as usize] gives you the corresponding token
        println!("  {:<15}  (distance = {:.4})", words[id as usize], dist);
    }
}
