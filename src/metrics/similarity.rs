//! This submodule provides some basic similarity measures
//!
//! It supports sparse vectors from `sprs` which seems to be the most popular
//! library for sparse algebra.

use sprs::CsVecOwned;

/// Type for a similarity function
pub type Similarity = fn(&CsVecOwned<f64>, &CsVecOwned<f64>) -> f64;

/// Cosine similarity between two vectors.
///
/// Returns zero if one of the vectors is zero.
pub fn cosine(a: &CsVecOwned<f64>, b: &CsVecOwned<f64>) -> f64 {
    let norms = a.dot(a) * b.dot(b);
    if norms > 0.0 {
        a.dot(b) / norms.sqrt()
    } else {
        0.0
    }
}

/// Jaccard similarity between two vectors.
///
/// Returns zero if one of the vectors is zero.
///
/// # Remarks
///
/// Jaccard similarity is only defined for positive values.
/// Using negative values will give an incorrect result.
pub fn jaccard(a: &CsVecOwned<f64>, b: &CsVecOwned<f64>) -> f64 {
    if a.nnz() == 0 || b.nnz() == 0 {
        return 0.;
    }

    let mut min_sum = 0.;
    let mut max_sum = 0.;

    for (index, value) in a.iter() {
        let value = *value;
        let other_value = b.get(index).cloned().unwrap_or(0.);
        if value < other_value {
            min_sum += value;
            max_sum += other_value;
        } else {
            min_sum += other_value;
            max_sum += value;
        }
    }

    if !(min_sum > 0.) {
        return 0.;
    }

    for (index, value) in b.iter() {
        if a.get(index).is_none() {
            max_sum += *value;
        }
    }

    min_sum / max_sum
}
