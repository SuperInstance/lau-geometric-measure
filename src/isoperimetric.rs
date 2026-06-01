//! Isoperimetric inequalities for geometric measure theory.
//!
//! The isoperimetric inequality states that for a k-dimensional surface with
//! boundary, the area is bounded below by a constant times the boundary length
//! raised to the appropriate power.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::currents::Current;
use crate::hausdorff::Point;

/// Result of an isoperimetric inequality check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsoperimetricCheck {
    /// Surface area.
    pub area: f64,
    /// Boundary measure.
    pub boundary_measure: f64,
    /// Isoperimetric constant used.
    pub constant: f64,
    /// Whether the inequality holds.
    pub holds: bool,
    /// Ratio area / lower_bound (≥ 1 if inequality holds).
    pub ratio: f64,
}

/// Classical isoperimetric constant C(k) for R^k.
///
/// For k=2: C = 1/(4π), so Area ≥ perimeter²/(4π).
pub fn isoperimetric_constant(k: usize) -> f64 {
    match k {
        1 => 0.5,
        2 => 1.0 / (4.0 * std::f64::consts::PI),
        3 => (36.0 * std::f64::consts::PI).powf(1.0 / 3.0),
        _ => {
            let n = k as f64;
            let omega_n = crate::hausdorff::volume_unit_ball(n);
            let omega_n1 = crate::hausdorff::volume_unit_ball(n - 1.0);
            if omega_n < 1e-15 { 1.0 } else {
                n.powf(n / (n - 1.0)) * omega_n1.powf(n / (n - 1.0)) / omega_n
            }
        }
    }
}

/// Check the isoperimetric inequality for a current.
///
/// For a k-current T with boundary ∂T:
/// M(T) ≥ C(k) · M(∂T)^(k/(k-1))
pub fn check_isoperimetric(current: &Current, k: usize) -> IsoperimetricCheck {
    let area = current.mass();
    let bdry = current.boundary().mass();
    let c = isoperimetric_constant(k);

    let lower_bound = if k >= 2 && bdry > 1e-10 {
        c * bdry.powf(k as f64 / (k - 1) as f64)
    } else {
        0.0
    };

    let ratio = if lower_bound > 1e-15 { area / lower_bound } else { f64::INFINITY };
    let holds = ratio >= 1.0 || lower_bound < 1e-10;

    IsoperimetricCheck {
        area,
        boundary_measure: bdry,
        constant: c,
        holds,
        ratio,
    }
}

/// Sobolev inequality constant for R^n.
///
/// For n > k: ||f||_{L^{n/(n-k)}} ≤ C · ||∇f||_{L^k}
pub fn sobolev_constant(n: usize, k: usize) -> f64 {
    if n <= k { return 0.0; }
    let omega_n = crate::hausdorff::volume_unit_ball(n as f64);
    let nk = n - k;
    n as f64 * omega_n.powf(1.0 / n as f64) / (nk as f64)
}

/// Cheeger constant for a domain.
///
/// h(Ω) = inf |∂A| / min(|A|, |Ω\A|) over all subsets A.
/// We approximate this for a discrete point cloud.
pub fn cheeger_constant(points: &[Point], k: usize) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    // Approximate by computing the isoperimetric profile
    let n = points.len();
    let mut best_ratio = f64::INFINITY;

    // Sample partitions
    for dim in 0..points[0].len() {
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            points[a][dim].partial_cmp(&points[b][dim]).unwrap()
        });

        for split in 1..n {
            let a_size = split;
            let b_size = n - split;
            let min_size = a_size.min(b_size) as f64;

            // Boundary: count pairs (i,j) where i in A, j in B, close together
            let mut boundary = 0.0_f64;
            for &i in &sorted_indices[..split] {
                for &j in &sorted_indices[split..] {
                    let d = (&points[i] - &points[j]).norm();
                    if d < 0.5 {
                        boundary += 1.0;
                    }
                }
            }

            if boundary > 0.0 {
                let ratio = boundary / min_size;
                if ratio < best_ratio {
                    best_ratio = ratio;
                }
            }
        }
    }

    if best_ratio.is_infinite() { 0.0 } else { best_ratio }
}

/// Compute the isoperimetric profile: minimum boundary for each volume.
pub fn isoperimetric_profile(points: &[Point], k: usize) -> Vec<(f64, f64)> {
    let n = points.len();
    if n == 0 { return vec![]; }

    let mut profile = Vec::new();

    for fraction in &[0.1, 0.2, 0.3, 0.4, 0.5] {
        let target = (*fraction * n as f64).ceil() as usize;
        let min_size = target.min(n - target);

        // Simple approximation: find the tightest cut
        let mut min_boundary = f64::INFINITY;
        for dim in 0..points[0].len() {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| points[a][dim].partial_cmp(&points[b][dim]).unwrap());

            let mut boundary = 0.0;
            for &i in &indices[..target.min(n)] {
                for &j in &indices[target.min(n)..] {
                    let d = (&points[i] - &points[j]).norm();
                    if d < 1.0 {
                        boundary += 1.0;
                    }
                }
            }
            if boundary < min_boundary {
                min_boundary = boundary;
            }
        }

        let volume = min_size as f64 / n as f64;
        profile.push((volume, if min_boundary.is_infinite() { 0.0 } else { min_boundary }));
    }

    profile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currents::Simplex;

    fn pt(x: f64, y: f64) -> Point {
        DVector::from_vec(vec![x, y])
    }

    #[test]
    fn test_isoperimetric_constant() {
        let c2 = isoperimetric_constant(2);
        assert!((c2 - 1.0 / (4.0 * std::f64::consts::PI)).abs() < 1e-10);
        let c1 = isoperimetric_constant(1);
        assert!((c1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_check_isoperimetric_disk() {
        // Triangulated disk
        let mut disk = Current::zero();
        let center = pt(0.5, 0.5);
        let n = 20;
        let r = 0.5;
        for i in 0..n {
            let a1 = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let a2 = 2.0 * std::f64::consts::PI * (i + 1) as f64 / n as f64;
            disk.add(Simplex::new(vec![
                center.clone(),
                pt(0.5 + r * a1.cos(), 0.5 + r * a1.sin()),
                pt(0.5 + r * a2.cos(), 0.5 + r * a2.sin()),
            ], 1.0));
        }
        let check = check_isoperimetric(&disk, 2);
        assert!(check.area > 0.0);
        assert!(check.ratio > 0.0);
    }

    #[test]
    fn test_check_isoperimetric_zero_current() {
        let c = Current::zero();
        let check = check_isoperimetric(&c, 2);
        assert_eq!(check.area, 0.0);
        assert!(check.holds);
    }

    #[test]
    fn test_sobolev_constant() {
        let s = sobolev_constant(3, 1);
        assert!(s > 0.0);
        let s0 = sobolev_constant(1, 1);
        assert_eq!(s0, 0.0);
    }

    #[test]
    fn test_cheeger_constant_grid() {
        let pts: Vec<Point> = (0..5)
            .flat_map(|i| (0..5).map(move |j| pt(i as f64, j as f64)))
            .collect();
        let h = cheeger_constant(&pts, 2);
        assert!(h >= 0.0);
    }

    #[test]
    fn test_cheeger_constant_empty() {
        let pts: Vec<Point> = vec![];
        let h = cheeger_constant(&pts, 1);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_isoperimetric_profile() {
        let pts: Vec<Point> = (0..10)
            .flat_map(|i| (0..10).map(move |j| pt(i as f64, j as f64)))
            .collect();
        let profile = isoperimetric_profile(&pts, 2);
        assert!(!profile.is_empty());
        assert_eq!(profile.len(), 5);
    }

    #[test]
    fn test_isoperimetric_profile_empty() {
        let pts: Vec<Point> = vec![];
        let profile = isoperimetric_profile(&pts, 1);
        assert!(profile.is_empty());
    }

    #[test]
    fn test_isoperimetric_check_serialization() {
        let check = IsoperimetricCheck {
            area: 1.0,
            boundary_measure: 2.0,
            constant: 0.5,
            holds: true,
            ratio: 1.5,
        };
        let json = serde_json::to_string(&check).unwrap();
        let c2: IsoperimetricCheck = serde_json::from_str(&json).unwrap();
        assert!(c2.holds);
        assert!((c2.ratio - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_isoperimetric_triangle() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let c = crate::currents::Current::from_simplex(tri);
        let check = check_isoperimetric(&c, 2);
        assert!(check.area > 0.0);
    }

    #[test]
    fn test_isoperimetric_constant_higher_dims() {
        for k in 1..=5 {
            let c = isoperimetric_constant(k);
            assert!(c > 0.0, "Constant for k={} should be positive", k);
        }
    }

    #[test]
    fn test_cheeger_single_point() {
        let pts = vec![pt(0.0, 0.0)];
        let h = cheeger_constant(&pts, 1);
        assert_eq!(h, 0.0);
    }
}
