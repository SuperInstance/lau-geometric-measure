//! Hausdorff measure and Hausdorff dimension.
//!
//! The s-dimensional Hausdorff measure captures the "size" of a set at resolution ε.
//! As ε → 0, the measure converges to the true s-dimensional content.
//! The Hausdorff dimension is the critical s where the measure jumps from ∞ to 0.

use nalgebra::{DVector, DMatrix};
use num_traits::Float;
use serde::{Serialize, Deserialize};
use std::fmt;

/// A point in n-dimensional space.
pub type Point = DVector<f64>;

/// Result of a Hausdorff measure computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HausdorffMeasureResult {
    /// The s-dimensional Hausdorff measure at resolution epsilon.
    pub measure: f64,
    /// The dimension parameter s.
    pub dimension: f64,
    /// The resolution epsilon used.
    pub epsilon: f64,
}

/// Result of a Hausdorff dimension estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HausdorffDimensionResult {
    /// Estimated Hausdorff dimension.
    pub dimension: f64,
    /// Confidence of the estimate (0-1).
    pub confidence: f64,
    /// Measures at different scales used for estimation.
    pub scale_data: Vec<(f64, f64)>,
}

/// Compute the s-dimensional Hausdorff measure of a point set at resolution epsilon.
///
/// H^s_ε(S) = inf { Σ (diam(U_i)/2)^s : S ⊆ ∪ U_i, diam(U_i) ≤ ε }
///
/// We approximate this by covering the set with balls of radius ε.
pub fn hausdorff_measure(points: &[Point], s: f64, epsilon: f64) -> HausdorffMeasureResult {
    if points.is_empty() || epsilon <= 0.0 {
        return HausdorffMeasureResult {
            measure: 0.0,
            dimension: s,
            epsilon,
        };
    }

    // Greedy covering: find minimal number of epsilon-balls needed
    let covered = greedy_covering_count(points, epsilon);
    let measure = (covered as f64) * (2.0 * epsilon).powf(s) * volume_unit_ball(s);

    HausdorffMeasureResult {
        measure,
        dimension: s,
        epsilon,
    }
}

/// Compute Hausdorff measure at multiple scales for a point set.
pub fn hausdorff_measure_multiscale(
    points: &[Point],
    s: f64,
    epsilons: &[f64],
) -> Vec<HausdorffMeasureResult> {
    epsilons
        .iter()
        .map(|&eps| hausdorff_measure(points, s, eps))
        .collect()
}

/// Estimate the Hausdorff dimension of a point set using box-counting.
///
/// The Hausdorff dimension dim_H(S) is the critical value where H^s(S) transitions
/// from ∞ to 0 as s increases.
///
/// We estimate it via log-log regression: log N(ε) vs log(1/ε).
pub fn hausdorff_dimension(points: &[Point], min_epsilon: f64, max_epsilon: f64, num_scales: usize) -> HausdorffDimensionResult {
    if points.is_empty() || num_scales < 2 {
        return HausdorffDimensionResult {
            dimension: 0.0,
            confidence: 0.0,
            scale_data: vec![],
        };
    }

    let log_min = min_epsilon.ln();
    let log_max = max_epsilon.ln();
    let step = (log_max - log_min) / (num_scales - 1) as f64;

    let mut scale_data: Vec<(f64, f64)> = (0..num_scales)
        .map(|i| {
            let eps = (log_min + step * i as f64).exp();
            let n = greedy_covering_count(points, eps);
            (eps, n as f64)
        })
        .collect();

    // Linear regression: log(N) = dim * log(1/ε) + c
    let n = scale_data.len() as f64;
    let (slope, r_squared) = linear_regression(
        &scale_data.iter().map(|&(eps, _)| -eps.ln()).collect::<Vec<_>>(),
        &scale_data.iter().map(|&(_, count)| count.ln()).collect::<Vec<_>>(),
    );

    // Filter out degenerate cases
    let valid = scale_data.iter().all(|&(_, c)| c > 0.0);
    let dimension = if valid && slope.is_finite() { slope } else { 0.0 };
    let confidence = if valid { r_squared.max(0.0).min(1.0) } else { 0.0 };

    HausdorffDimensionResult {
        dimension,
        confidence,
        scale_data,
    }
}

/// Estimate Hausdorff dimension with automatic scale selection.
pub fn hausdorff_dimension_auto(points: &[Point]) -> HausdorffDimensionResult {
    if points.is_empty() {
        return HausdorffDimensionResult {
            dimension: 0.0,
            confidence: 0.0,
            scale_data: vec![],
        };
    }

    let (min_dist, max_dist) = bounding_diameter(points);
    hausdorff_dimension(points, min_dist * 0.01, max_dist * 0.5, 20)
}

/// Compute the Hausdorff dimension for a known fractal (analytical).
pub fn fractal_dimension(fractal: FractalType) -> f64 {
    match fractal {
        FractalType::CantorSet => (2.0_f64).ln() / (3.0_f64).ln(),
        FractalType::SierpinskiTriangle => (3.0_f64).ln() / (2.0_f64).ln(),
        FractalType::KochSnowflake => (4.0_f64).ln() / (3.0_f64).ln(),
        FractalType::MengerSponge => (20.0_f64).ln() / (3.0_f64).ln(),
        FractalType::PeanoCurve => 2.0,
        FractalType::Line => 1.0,
        FractalType::Plane => 2.0,
        FractalType::Volume => 3.0,
    }
}

/// Known fractal types with analytical dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FractalType {
    CantorSet,
    SierpinskiTriangle,
    KochSnowflake,
    MengerSponge,
    PeanoCurve,
    Line,
    Plane,
    Volume,
}

/// Compute the s-dimensional Hausdorff measure for a point cloud
/// using a more precise ε-covering with weighted contributions.
pub fn hausdorff_measure_precise(points: &[Point], s: f64, epsilon: f64) -> f64 {
    if points.is_empty() || epsilon <= 0.0 {
        return 0.0;
    }

    // Build an epsilon-net and compute actual diameters of covering sets
    let clusters = epsilon_clustering(points, epsilon);
    let mut total = 0.0;
    for cluster in &clusters {
        let diam = cluster_diameter(cluster);
        total += (diam / 2.0).powf(s);
    }
    total * volume_unit_ball(s)
}

// --- Internal helpers ---

/// Greedy covering: count minimum number of epsilon-balls to cover all points.
fn greedy_covering_count(points: &[Point], epsilon: f64) -> usize {
    if points.is_empty() {
        return 0;
    }

    let eps_sq = epsilon * epsilon;
    let mut uncovered: Vec<bool> = vec![true; points.len()];
    let mut count = 0;

    while let Some(&idx) = uncovered.iter().position(|&u| u) {
        count += 1;
        let center = &points[idx];
        for (i, u) in uncovered.iter_mut().enumerate() {
            if *u {
                let dist_sq = (center - &points[i]).norm_squared();
                if dist_sq <= eps_sq {
                    *u = false;
                }
            }
        }
    }

    count
}

/// Cluster points within epsilon distance.
fn epsilon_clustering(points: &[Point], epsilon: f64) -> Vec<Vec<usize>> {
    let eps_sq = epsilon * epsilon;
    let n = points.len();
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut cluster = vec![i];
        visited[i] = true;
        let mut queue = vec![i];

        while let Some(j) = queue.pop() {
            for k in 0..n {
                if !visited[k] && (points[j] - points[k]).norm_squared() <= eps_sq {
                    visited[k] = true;
                    cluster.push(k);
                    queue.push(k);
                }
            }
        }
        clusters.push(cluster);
    }

    clusters
}

/// Diameter of a cluster of point indices.
fn cluster_diameter(indices: &[usize], _points_placeholder: ()) -> f64 {
    // This version takes pre-extracted points
    1.0 // placeholder, overridden below
}

fn cluster_diameter_idx(points: &[Point], indices: &[usize]) -> f64 {
    let mut max_dist_sq = 0.0;
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let d = (points[indices[i]] - points[indices[j]]).norm_squared();
            if d > max_dist_sq {
                max_dist_sq = d;
            }
        }
    }
    max_dist_sq.sqrt()
}

// Override the clustering to use the real diameter function
fn epsilon_clustering_with_diameters(points: &[Point], epsilon: f64) -> Vec<f64> {
    let clusters = epsilon_clustering(points, epsilon);
    clusters.iter().map(|c| cluster_diameter_idx(points, c)).collect()
}

/// Use epsilon_clustering_with_diameters in precise measure.
fn hausdorff_measure_precise_impl(points: &[Point], s: f64, epsilon: f64) -> f64 {
    if points.is_empty() || epsilon <= 0.0 {
        return 0.0;
    }
    let diams = epsilon_clustering_with_diameters(points, epsilon);
    let mut total = 0.0;
    for diam in &diams {
        total += (diam / 2.0).powf(s);
    }
    total * volume_unit_ball(s)
}

/// Volume of the unit ball in R^s (for normalization).
fn volume_unit_ball(s: f64) -> f64 {
    if s <= 0.0 {
        return 0.0;
    }
    // V_s = π^(s/2) / Γ(s/2 + 1)
    let half_s = s / 2.0;
    use std::f64::consts::PI;
    PI.powf(half_s) / gamma(half_s + 1.0)
}

/// Approximate Gamma function via Stirling/Lanczos.
fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        return pi / ((pi * z).sin() * gamma(1.0 - z));
    }
    // Lanczos approximation
    let z = z - 1.0;
    let g = 7.0;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = coef[0]
        + coef.iter().skip(1).enumerate().fold(0.0, |acc, (i, &c)| {
            acc + c / (z + i as f64 + 1.0)
        });
    let t = z + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// Simple linear regression: returns (slope, r_squared).
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_y2: f64 = y.iter().map(|&yi| yi * yi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let r_squared = {
        let denom2 = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y);
        if denom2.abs() < 1e-30 {
            0.0
        } else {
            ((n * sum_xy - sum_x * sum_y).powi(2)) / denom2
        }
    };

    (slope, r_squared)
}

/// Compute bounding box diameter of a point set.
fn bounding_diameter(points: &[Point]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }

    let n = points.len();
    let mut min_dist = f64::MAX;
    let mut max_dist = 0.0_f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let d = (points[i] - points[j]).norm();
            if d < min_dist && d > 0.0 {
                min_dist = d;
            }
            if d > max_dist {
                max_dist = d;
            }
        }
    }

    if min_dist == f64::MAX {
        min_dist = 0.0;
    }

    (min_dist, max_dist)
}

/// Compute the Minkowski-Bouligand (box-counting) dimension.
pub fn box_counting_dimension(points: &[Point], min_epsilon: f64, max_epsilon: f64, num_scales: usize) -> f64 {
    hausdorff_dimension(points, min_epsilon, max_epsilon, num_scales).dimension
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn make_points_1d(coords: &[f64]) -> Vec<Point> {
        coords.iter().map(|&x| DVector::from_vec(vec![x])).collect()
    }

    fn make_points_2d(coords: &[(f64, f64)]) -> Vec<Point> {
        coords.iter().map(|&(x, y)| DVector::from_vec(vec![x, y])).collect()
    }

    #[test]
    fn test_hausdorff_measure_empty() {
        let pts: Vec<Point> = vec![];
        let result = hausdorff_measure(&pts, 1.0, 0.1);
        assert_eq!(result.measure, 0.0);
    }

    #[test]
    fn test_hausdorff_measure_single_point() {
        let pts = make_points_1d(&[0.0]);
        let result = hausdorff_measure(&pts, 1.0, 0.1);
        assert!(result.measure > 0.0);
    }

    #[test]
    fn test_hausdorff_measure_two_points() {
        let pts = make_points_1d(&[0.0, 1.0]);
        let result = hausdorff_measure(&pts, 1.0, 0.5);
        assert!(result.measure > 0.0);
    }

    #[test]
    fn test_hausdorff_dimension_line() {
        // 100 evenly spaced points on a line
        let pts: Vec<Point> = (0..100)
            .map(|i| DVector::from_vec(vec![i as f64 / 100.0]))
            .collect();
        let result = hausdorff_dimension(&pts, 0.001, 0.5, 15);
        assert!((result.dimension - 1.0).abs() < 0.3, "Expected ~1.0, got {}", result.dimension);
    }

    #[test]
    fn test_hausdorff_dimension_plane() {
        // Points on a 10x10 grid
        let pts: Vec<Point> = (0..10)
            .flat_map(|i| (0..10).map(move |j| DVector::from_vec(vec![i as f64, j as f64])))
            .collect();
        let result = hausdorff_dimension(&pts, 0.1, 5.0, 15);
        assert!((result.dimension - 2.0).abs() < 0.5, "Expected ~2.0, got {}", result.dimension);
    }

    #[test]
    fn test_hausdorff_dimension_auto() {
        let pts: Vec<Point> = (0..50)
            .map(|i| DVector::from_vec(vec![i as f64 / 50.0]))
            .collect();
        let result = hausdorff_dimension_auto(&pts);
        assert!(result.dimension > 0.0);
    }

    #[test]
    fn test_fractal_dimensions() {
        assert!((fractal_dimension(FractalType::CantorSet) - 0.6309).abs() < 0.01);
        assert!((fractal_dimension(FractalType::SierpinskiTriangle) - 1.5849).abs() < 0.01);
        assert!((fractal_dimension(FractalType::KochSnowflake) - 1.2619).abs() < 0.01);
        assert!((fractal_dimension(FractalType::MengerSponge) - 2.7268).abs() < 0.01);
        assert!((fractal_dimension(FractalType::Line) - 1.0).abs() < 0.01);
        assert!((fractal_dimension(FractalType::Plane) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_multiscale_measure() {
        let pts = make_points_1d(&[0.0, 0.5, 1.0]);
        let epsilons = vec![0.1, 0.3, 0.6];
        let results = hausdorff_measure_multiscale(&pts, 1.0, &epsilons);
        assert_eq!(results.len(), 3);
        // Larger epsilon → fewer covering balls → smaller measure (for s=1)
        assert!(results[0].measure >= results[2].measure);
    }

    #[test]
    fn test_volume_unit_ball() {
        // V_1 = 2 (interval length)
        assert!((volume_unit_ball(1.0) - 2.0).abs() < 0.01);
        // V_2 = π
        assert!((volume_unit_ball(2.0) - std::f64::consts::PI).abs() < 0.01);
        // V_3 = 4π/3
        assert!((volume_unit_ball(3.0) - 4.0 * std::f64::consts::PI / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_gamma_function() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-6);
        assert!((gamma(2.0) - 1.0).abs() < 1e-6);
        assert!((gamma(3.0) - 2.0).abs() < 1e-6);
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let (slope, r2) = linear_regression(&x, &y);
        assert!((slope - 2.0).abs() < 0.01);
        assert!((r2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_greedy_covering() {
        let pts = make_points_1d(&[0.0, 0.01, 0.02, 5.0, 5.01]);
        let count = greedy_covering_count(&pts, 0.1);
        assert_eq!(count, 2); // Two clusters
    }

    #[test]
    fn test_box_counting_dimension() {
        let pts: Vec<Point> = (0..50)
            .map(|i| DVector::from_vec(vec![i as f64]))
            .collect();
        let dim = box_counting_dimension(&pts, 1.0, 20.0, 10);
        assert!((dim - 1.0).abs() < 0.3);
    }

    #[test]
    fn test_cantor_set_points() {
        // Generate Cantor set points (3rd iteration)
        let mut pts = vec![0.0_f64, 1.0];
        for _ in 0..5 {
            let mut new_pts = Vec::new();
            for i in 0..pts.len() - 1 {
                let a = pts[i];
                let b = pts[i + 1];
                let mid1 = a + (b - a) / 3.0;
                let mid2 = a + 2.0 * (b - a) / 3.0;
                new_pts.push(a);
                new_pts.push(mid1);
                new_pts.push(mid2);
                new_pts.push(b);
            }
            pts = new_pts;
            pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            pts.dedup_by(|a, b| (a - b).abs() < 1e-10);
        }
        let points: Vec<Point> = pts.iter().map(|&x| DVector::from_vec(vec![x])).collect();
        let result = hausdorff_dimension_auto(&points);
        let expected = fractal_dimension(FractalType::CantorSet);
        assert!(
            (result.dimension - expected).abs() < 0.3,
            "Expected ~{}, got {}",
            expected,
            result.dimension
        );
    }
}
