//! Rectifiability — when can a set be covered by Lipschitz images of R^n?
//!
//! A set is k-rectifiable if it can be covered (up to H^k-null set) by
//! countably many Lipschitz images of R^k. This is crucial for understanding
//! when agent manifolds have nice geometric structure.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::hausdorff::Point;

/// Rectifiability classification of a set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Rectifiability {
    /// k-rectifiable: covered by countably many Lipschitz images of R^k.
    Rectifiable { dimension: usize },
    /// Purely k-unrectifiable: contains no rectifiable subset of positive H^k measure.
    Unrectifiable { dimension: usize },
    /// Mixed: partially rectifiable.
    Partial { dimension: usize, rectifiable_fraction: f64 },
}

/// Result of a rectifiability test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RectifiabilityResult {
    /// Classification.
    pub classification: Rectifiability,
    /// The tangent plane quality: how well local patches align.
    pub tangent_quality: f64,
    /// Approximate dimension of the set.
    pub estimated_dimension: f64,
}

/// Test rectifiability of a point cloud by checking approximate tangent planes.
///
/// A set is k-rectifiable iff it has an approximate k-dimensional tangent plane
/// at H^k-almost every point. We test this by:
/// 1. Computing local PCA at sample points
/// 2. Checking if the top-k eigenvalues dominate
/// 3. Measuring consistency of tangent directions
pub fn test_rectifiability(
    points: &[Point],
    k: usize,
    neighborhood_radius: f64,
    num_samples: usize,
) -> RectifiabilityResult {
    if points.is_empty() {
        return RectifiabilityResult {
            classification: Rectifiability::Rectifiable { dimension: 0 },
            tangent_quality: 1.0,
            estimated_dimension: 0.0,
        };
    }

    let sample_count = num_samples.min(points.len());
    let step = (points.len() as f64 / sample_count as f64).ceil() as usize;
    let sample_indices: Vec<usize> = (0..points.len()).step_by(step.max(1)).take(sample_count).collect();

    let mut tangent_qualities = Vec::new();
    let mut dimension_estimates = Vec::new();

    for &idx in &sample_indices {
        let neighbors = find_neighbors(points, idx, neighborhood_radius);
        if neighbors.len() < k + 1 {
            continue;
        }

        let (eigenvalues, _eigenvectors) = local_pca(points, &neighbors);
        if eigenvalues.is_empty() {
            continue;
        }

        // Quality: ratio of top-k eigenvalue sum to total
        let total: f64 = eigenvalues.iter().sum();
        if total < 1e-15 {
            continue;
        }
        let top_k_sum: f64 = eigenvalues.iter().take(k).sum();
        tangent_qualities.push(top_k_sum / total);

        // Estimate local dimension via eigenvalue gap
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut dim_est = 0;
        let threshold = sorted.first().copied().unwrap_or(0.0) * 0.01;
        for &ev in &sorted {
            if ev > threshold {
                dim_est += 1;
            }
        }
        dimension_estimates.push(dim_est as f64);
    }

    if tangent_qualities.is_empty() {
        return RectifiabilityResult {
            classification: Rectifiability::Unrectifiable { dimension: k },
            tangent_quality: 0.0,
            estimated_dimension: 0.0,
        };
    }

    let avg_quality: f64 = tangent_qualities.iter().sum::<f64>() / tangent_qualities.len() as f64;
    let avg_dim: f64 = dimension_estimates.iter().sum::<f64>() / dimension_estimates.len() as f64;

    let classification = if avg_quality > 0.9 {
        Rectifiability::Rectifiable { dimension: k }
    } else if avg_quality < 0.3 {
        Rectifiability::Unrectifiable { dimension: k }
    } else {
        Rectifiability::Partial {
            dimension: k,
            rectifiable_fraction: avg_quality,
        }
    };

    RectifiabilityResult {
        classification,
        tangent_quality: avg_quality,
        estimated_dimension: avg_dim,
    }
}

/// Check if a point cloud lies on a Lipschitz graph over R^k.
///
/// Returns (is_lipschitz, estimated_lipschitz_constant).
pub fn check_lipschitz_graph(
    points: &[Point],
    k: usize,
    graph_direction: &DVector<f64>,
) -> (bool, f64) {
    if points.len() < 2 {
        return (true, 0.0);
    }

    let n = points[0].len();
    if k >= n {
        return (false, f64::INFINITY);
    }

    // Project onto graph_direction and perpendicular subspace
    let mut lipschitz_ratios = Vec::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let diff = &points[i] - &points[j];
            let proj_parallel = diff.dot(graph_direction);
            let proj_perp_sq = diff.norm_squared() - proj_parallel * proj_parallel;
            if proj_parallel.abs() > 1e-10 {
                let ratio = (proj_perp_sq.max(0.0)).sqrt() / proj_parallel.abs();
                lipschitz_ratios.push(ratio);
            }
        }
    }

    if lipschitz_ratios.is_empty() {
        return (true, 0.0);
    }

    let max_ratio = lipschitz_ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let is_lipschitz = max_ratio.is_finite();

    (is_lipschitz, max_ratio)
}

/// Decompose a set into rectifiable and unrectifiable parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RectifiableDecomposition {
    /// Points classified as rectifiable.
    pub rectifiable_points: Vec<usize>,
    /// Points classified as unrectifiable.
    pub unrectifiable_points: Vec<usize>,
    /// Fraction of points that are rectifiable.
    pub rectifiable_fraction: f64,
}

/// Decompose point cloud into rectifiable and unrectifiable parts.
pub fn decompose_rectifiability(
    points: &[Point],
    k: usize,
    neighborhood_radius: f64,
) -> RectifiableDecomposition {
    let threshold = 0.7;
    let mut rectifiable = Vec::new();
    let mut unrectifiable = Vec::new();

    for idx in 0..points.len() {
        let neighbors = find_neighbors(points, idx, neighborhood_radius);
        if neighbors.len() < k + 1 {
            unrectifiable.push(idx);
            continue;
        }

        let (eigenvalues, _) = local_pca(points, &neighbors);
        let total: f64 = eigenvalues.iter().sum();
        if total < 1e-15 {
            unrectifiable.push(idx);
            continue;
        }
        let top_k: f64 = eigenvalues.iter().take(k).sum();
        let quality = top_k / total;

        if quality > threshold {
            rectifiable.push(idx);
        } else {
            unrectifiable.push(idx);
        }
    }

    let frac = if points.is_empty() {
        0.0
    } else {
        rectifiable.len() as f64 / points.len() as f64
    };

    RectifiableDecomposition {
        rectifiable_points: rectifiable,
        unrectifiable_points: unrectifiable,
        rectifiable_fraction: frac,
    }
}

/// Compute the density of a rectifiable set at a point.
/// For a k-rectifiable set, the density is 1 at H^k-almost every point.
pub fn compute_density(
    points: &[Point],
    idx: usize,
    radius: f64,
    k: usize,
) -> f64 {
    let neighbors = find_neighbors(points, idx, radius);
    let n = neighbors.len() as f64;
    let vol_unit = crate::hausdorff::volume_unit_ball(k as f64) * radius.powi(k as i32);
    if vol_unit < 1e-15 {
        return 0.0;
    }
    n / vol_unit
}

// --- Internal helpers ---

fn find_neighbors(points: &[Point], center_idx: usize, radius: f64) -> Vec<usize> {
    let r2 = radius * radius;
    let center = &points[center_idx];
    points
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            *i != center_idx && (*p - center).norm_squared() <= r2
        })
        .map(|(i, _)| i)
        .collect()
}

/// Local PCA: returns (eigenvalues, eigenvectors) sorted ascending.
fn local_pca(points: &[Point], indices: &[usize]) -> (Vec<f64>, Vec<DVector<f64>>) {
    if indices.is_empty() {
        return (vec![], vec![]);
    }

    let dim = points[0].len();
    let n = indices.len() as f64;

    // Compute centroid
    let mut centroid = DVector::zeros(dim);
    for &idx in indices {
        centroid += &points[idx];
    }
    centroid /= n;

    // Compute covariance matrix
    let mut cov = vec![vec![0.0; dim]; dim];
    for &idx in indices {
        let diff = &points[idx] - &centroid;
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] += diff[i] * diff[j];
            }
        }
    }
    for row in cov.iter_mut() {
        for val in row.iter_mut() {
            *val /= n;
        }
    }

    // Simple eigenvalue decomposition for small matrices
    // Power iteration for top eigenvalues
    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();
    let mut deflated = cov.clone();

    for _ in 0..dim {
        let (eigenval, vec) = power_iteration(&deflated, dim, 100);
        eigenvalues.push(eigenval.max(0.0));
        eigenvectors.push(DVector::from_vec(vec));

        // Deflate
        let last_ev = eigenvectors.last().unwrap();
        for (i, row) in deflated.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell -= eigenval * last_ev[i] * last_ev[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

fn power_iteration(matrix: &[Vec<f64>], dim: usize, max_iter: usize) -> (f64, Vec<f64>) {
    let mut v = vec![1.0; dim];
    let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
    for x in &mut v {
        *x /= norm;
    }

    for _ in 0..max_iter {
        let mut new_v = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                new_v[i] += matrix[i][j] * v[j];
            }
        }
        let norm = (new_v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm < 1e-15 {
            return (0.0, v);
        }
        for x in &mut new_v {
            *x /= norm;
        }
        v = new_v;
    }

    // Compute eigenvalue: v^T A v
    let mut av = vec![0.0; dim];
    for i in 0..dim {
        for j in 0..dim {
            av[i] += matrix[i][j] * v[j];
        }
    }
    let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(&a, &b)| a * b).sum();

    (eigenvalue, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn line_points(n: usize) -> Vec<Point> {
        (0..n)
            .map(|i| DVector::from_vec(vec![i as f64 / (n as f64 - 1.0), 0.0]))
            .collect()
    }

    fn plane_points(n: usize) -> Vec<Point> {
        let side = (n as f64).sqrt() as usize;
        (0..side)
            .flat_map(|i| {
                (0..side).map(move |j| {
                    DVector::from_vec(vec![i as f64, j as f64, 0.0])
                })
            })
            .collect()
    }

    fn random_3d_points(n: usize) -> Vec<Point> {
        // Simulated random with simple hash for reproducibility
        (0..n)
            .map(|i| {
                let x = ((i * 7919 + 1) % 1000) as f64 / 1000.0;
                let y = ((i * 6271 + 3) % 1000) as f64 / 1000.0;
                let z = ((i * 3571 + 7) % 1000) as f64 / 1000.0;
                DVector::from_vec(vec![x, y, z])
            })
            .collect()
    }

    #[test]
    fn test_rectifiability_line() {
        let pts = line_points(50);
        let result = test_rectifiability(&pts, 1, 0.1, 10);
        assert_eq!(
            result.classification,
            Rectifiability::Rectifiable { dimension: 1 }
        );
        assert!(result.tangent_quality > 0.9);
    }

    #[test]
    fn test_rectifiability_plane() {
        let pts = plane_points(100);
        let result = test_rectifiability(&pts, 2, 2.0, 10);
        assert!(result.tangent_quality > 0.8);
    }

    #[test]
    fn test_rectifiability_random_3d() {
        let pts = random_3d_points(100);
        let result = test_rectifiability(&pts, 1, 0.5, 10);
        // Random 3D points should not be 1-rectifiable
        assert!(result.tangent_quality < 0.95);
    }

    #[test]
    fn test_lipschitz_graph() {
        // Points on a Lipschitz graph y = x
        let pts: Vec<Point> = (0..20)
            .map(|i| {
                let x = i as f64 / 20.0;
                DVector::from_vec(vec![x, x])
            })
            .collect();
        let dir = DVector::from_vec(vec![1.0, 0.0]);
        let (is_lip, constant) = check_lipschitz_graph(&pts, 1, &dir);
        assert!(is_lip);
        assert!(constant < 5.0);
    }

    #[test]
    fn test_decomposition_line() {
        let pts = line_points(30);
        let decomp = decompose_rectifiability(&pts, 1, 0.1);
        assert!(decomp.rectifiable_fraction > 0.5);
    }

    #[test]
    fn test_density_line() {
        let pts = line_points(100);
        // Density along a 1-rectifiable curve should be roughly constant
        let d = compute_density(&pts, 50, 0.2, 1);
        assert!(d > 0.0);
    }

    #[test]
    fn test_rectifiability_empty() {
        let pts: Vec<Point> = vec![];
        let result = test_rectifiability(&pts, 1, 0.1, 10);
        assert_eq!(
            result.classification,
            Rectifiability::Rectifiable { dimension: 0 }
        );
    }

    #[test]
    fn test_rectifiability_single_point() {
        let pts = vec![DVector::from_vec(vec![0.0, 0.0])];
        let result = test_rectifiability(&pts, 1, 0.1, 10);
        // Should not crash; classification depends on neighbor count
        assert!(result.estimated_dimension >= 0.0);
    }
}
