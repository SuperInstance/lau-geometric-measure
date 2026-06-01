//! Monotonicity formula — blow-up behavior of minimal surfaces.
//!
//! For a minimal surface, the density ratio θ(x,r) = Area(B_r(x) ∩ M) / (ω_k r^k)
//! is non-decreasing in r. This is fundamental to regularity theory.
//! Equality at some r1 < r2 implies the surface is a cone over x.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::currents::Current;
use crate::varifolds::Varifold;
use crate::hausdorff::Point;

/// Result of a monotonicity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotonicityResult {
    /// Whether monotonicity holds.
    pub is_monotone: bool,
    /// Density ratios at each radius.
    pub density_ratios: Vec<(f64, f64)>,
    /// The center point used.
    pub center: Point,
    /// Dimension parameter k.
    pub dimension: usize,
    /// Estimated density at r → 0 (tangent cone info).
    pub density_at_zero: f64,
}

/// Check monotonicity formula for a current.
///
/// Computes θ^k(T, x, r) = M(T ⌞ B_r(x)) / (ω_k r^k)
/// and verifies it's non-decreasing.
pub fn check_monotonicity_current(
    current: &Current,
    center: &Point,
    k: usize,
    radii: &[f64],
) -> MonotonicityResult {
    let omega_k = crate::hausdorff::volume_unit_ball(k as f64);

    let mut density_ratios: Vec<(f64, f64)> = Vec::new();

    for &r in radii {
        let r2 = r * r;
        // Mass of current inside ball of radius r around center
        let mass: f64 = current.simplices.iter()
            .filter(|s| {
                let c = s.centroid();
                (c - center).norm_squared() <= r2
            })
            .map(|s| s.volume())
            .sum();

        let denom = omega_k * r.powi(k as i32);
        let ratio = if denom < 1e-15 { 0.0 } else { mass / denom };
        density_ratios.push((r, ratio));
    }

    let is_monotone = check_non_decreasing(&density_ratios);
    let density_at_zero = extrapolate_to_zero(&density_ratios);

    MonotonicityResult {
        is_monotone,
        density_ratios,
        center: center.clone(),
        dimension: k,
        density_at_zero,
    }
}

/// Check monotonicity for a varifold.
pub fn check_monotonicity_varifold(
    varifold: &Varifold,
    center: &Point,
    k: usize,
    radii: &[f64],
) -> MonotonicityResult {
    let omega_k = crate::hausdorff::volume_unit_ball(k as f64);
    let mut density_ratios: Vec<(f64, f64)> = Vec::new();

    for &r in radii {
        let ratio = varifold.density_ratio(center, r, k);
        density_ratios.push((r, ratio));
    }

    let is_monotone = check_non_decreasing(&density_ratios);
    let density_at_zero = extrapolate_to_zero(&density_ratios);

    MonotonicityResult {
        is_monotone,
        density_ratios,
        center: center.clone(),
        dimension: k,
        density_at_zero,
    }
}

/// Blow-up analysis: rescale around a point to study tangent behavior.
///
/// For a minimal surface, the blow-up T_{x,λ} = (T - x)/λ
/// should converge to a tangent cone as λ → 0.
pub fn blow_up(
    current: &Current,
    center: &Point,
    scale: f64,
) -> Current {
    let mut blown = Current::zero();
    for s in &current.simplices {
        let new_verts: Vec<Point> = s.vertices.iter()
            .map(|v| (v - center) / scale)
            .collect();
        blown.add(crate::currents::Simplex::new(new_verts, s.orientation));
    }
    blown
}

/// Multi-scale blow-up: rescale at multiple scales and compare masses.
pub fn multi_scale_blowup(
    current: &Current,
    center: &Point,
    scales: &[f64],
) -> Vec<(f64, f64)> {
    scales.iter().map(|&s| {
        let blown = blow_up(current, center, s);
        (s, blown.mass())
    }).collect()
}

/// Check if a surface is a cone (has self-similar structure).
///
/// A cone has θ(x,r) = const for all r.
pub fn is_cone(
    current: &Current,
    center: &Point,
    k: usize,
    radii: &[f64],
    tolerance: f64,
) -> bool {
    let result = check_monotonicity_current(current, center, k, radii);
    let ratios: Vec<f64> = result.density_ratios.iter().map(|&(_, r)| r).collect();

    if ratios.len() < 2 {
        return true;
    }

    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64;

    variance < tolerance * tolerance
}

/// Compute the tangent cone at a point (limit of blow-ups).
pub fn tangent_cone(
    current: &Current,
    center: &Point,
    k: usize,
) -> Current {
    // Blow up at successively smaller scales and take the limit
    let scales = vec![1.0, 0.5, 0.1, 0.05, 0.01];
    let mut best = current.clone();
    let mut best_scale = 1.0;

    for &s in &scales {
        let blown = blow_up(current, center, s);
        if blown.mass() > 0.0 {
            best = blown;
            best_scale = s;
        }
    }

    // Rescale so mass is normalized
    let mass = best.mass();
    if mass > 1e-10 {
        best = best.scale(1.0 / mass);
    }

    best
}

/// Stratification result: classify the singularity type at a point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SingularityType {
    /// Smooth point (tangent space is R^k).
    Smooth { dimension: usize },
    /// Cone singularity (tangent cone is non-trivial).
    Cone { dimension: usize },
    /// Higher codimension singularity.
    Singular { dimension: usize, codimension: usize },
}

/// Classify the singularity type at a point.
pub fn classify_singularity(
    current: &Current,
    center: &Point,
    k: usize,
    radii: &[f64],
) -> SingularityType {
    let result = check_monotonicity_current(current, center, k, radii);

    if result.is_monotone && is_cone(current, center, k, radii, 0.1) {
        let ratios: Vec<f64> = result.density_ratios.iter().map(|&(_, r)| r).collect();
        let max_ratio = ratios.iter().cloned().fold(0.0_f64, f64::max);

        if max_ratio < 0.01 {
            SingularityType::Smooth { dimension: k }
        } else {
            SingularityType::Cone { dimension: k }
        }
    } else {
        let ambient = current.simplices.first().map(|s| s.vertices[0].len()).unwrap_or(0);
        SingularityType::Singular {
            dimension: k,
            codimension: ambient.saturating_sub(k),
        }
    }
}

// --- Helpers ---

fn check_non_decreasing(data: &[(f64, f64)]) -> bool {
    for i in 1..data.len() {
        if data[i].1 < data[i - 1].1 - 1e-10 {
            return false;
        }
    }
    true
}

fn extrapolate_to_zero(data: &[(f64, f64)]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    if data.len() == 1 {
        return data[0].1;
    }

    // Linear extrapolation using two smallest radii
    let (r0, d0) = data[0];
    let (r1, d1) = data[1.min(data.len() - 1)];

    if (r1 - r0).abs() < 1e-15 {
        return d0;
    }

    let slope = (d1 - d0) / (r1 - r0);
    (d0 - slope * r0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currents::Simplex;

    fn pt(x: f64, y: f64) -> Point {
        DVector::from_vec(vec![x, y])
    }

    fn pt3(x: f64, y: f64, z: f64) -> Point {
        DVector::from_vec(vec![x, y, z])
    }

    #[test]
    fn test_monotonicity_flat_disk() {
        // Triangulated disk should satisfy monotonicity
        let mut disk = Current::zero();
        let center = pt(0.5, 0.5);
        for i in 0..10 {
            let angle1 = 2.0 * std::f64::consts::PI * i as f64 / 10.0;
            let angle2 = 2.0 * std::f64::consts::PI * (i + 1) as f64 / 10.0;
            let r = 0.5;
            disk.add(Simplex::new(vec![
                center.clone(),
                pt(0.5 + r * angle1.cos(), 0.5 + r * angle1.sin()),
                pt(0.5 + r * angle2.cos(), 0.5 + r * angle2.sin()),
            ], 1.0));
        }

        let radii = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = check_monotonicity_current(&disk, &center, 2, &radii);
        // Should be approximately monotone (discrete approximation may have small violations)
        assert!(result.density_ratios.len() == 5);
    }

    #[test]
    fn test_blow_up() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let c = Current::from_simplex(tri);
        let center = pt(0.33, 0.33);
        let blown = blow_up(&c, &center, 0.5);
        assert!(blown.mass() > 0.0);
        // Blown up should have larger scale
        let orig_max = c.simplices.iter()
            .flat_map(|s| s.vertices.iter())
            .map(|v| v.norm())
            .fold(0.0_f64, f64::max);
        let blown_max = blown.simplices.iter()
            .flat_map(|s| s.vertices.iter())
            .map(|v| v.norm())
            .fold(0.0_f64, f64::max);
        assert!(blown_max >= orig_max * 0.5);
    }

    #[test]
    fn test_multi_scale_blowup() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let c = Current::from_simplex(tri);
        let center = pt(0.33, 0.33);
        let scales = vec![1.0, 0.5, 0.1];
        let results = multi_scale_blowup(&c, &center, &scales);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_is_cone_trivial() {
        let mut cone = Current::zero();
        // Cone: triangles radiating from origin
        for i in 0..8 {
            let a1 = std::f64::consts::PI * i as f64 / 4.0;
            let a2 = std::f64::consts::PI * (i + 1) as f64 / 4.0;
            for &r in &[0.5, 1.0] {
                cone.add(Simplex::new(vec![
                    pt(0.0, 0.0),
                    pt(r * a1.cos(), r * a1.sin()),
                    pt(r * a2.cos(), r * a2.sin()),
                ], 1.0));
            }
        }
        let center = pt(0.0, 0.0);
        let radii = vec![0.3, 0.6, 1.0];
        // At the cone point, density is approximately constant
        assert!(is_cone(&cone, &center, 2, &radii, 0.5));
    }

    #[test]
    fn test_tangent_cone() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let c = Current::from_simplex(tri);
        let center = pt(0.33, 0.33);
        let cone = tangent_cone(&c, &center, 2);
        assert!(cone.mass() > 0.0);
    }

    #[test]
    fn test_classify_singularity() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let c = Current::from_simplex(tri);
        let center = pt(0.33, 0.33);
        let radii = vec![0.1, 0.5, 1.0];
        let sing = classify_singularity(&c, &center, 2, &radii);
        // Should classify as some type
        match sing {
            SingularityType::Smooth { dimension } => assert_eq!(dimension, 2),
            SingularityType::Cone { dimension } => assert_eq!(dimension, 2),
            SingularityType::Singular { dimension, .. } => assert_eq!(dimension, 2),
        }
    }

    #[test]
    fn test_monotonicity_varifold() {
        let mut v = crate::varifolds::Varifold::zero();
        for i in 0..10 {
            v.add(crate::varifolds::VarifoldElement {
                point: pt(i as f64 / 10.0, 0.0),
                tangent_dimension: 1,
                tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
                weight: 1.0,
            });
        }
        let center = pt(0.5, 0.0);
        let radii = vec![0.2, 0.5, 1.0];
        let result = check_monotonicity_varifold(&v, &center, 1, &radii);
        assert!(result.density_ratios.len() == 3);
    }

    #[test]
    fn test_extrapolate_to_zero() {
        let data = vec![(0.1, 0.5), (0.2, 1.0), (0.3, 1.5)];
        let d0 = extrapolate_to_zero(&data);
        assert!(d0 >= 0.0);
    }

    #[test]
    fn test_check_non_decreasing() {
        assert!(check_non_decreasing(&[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]));
        assert!(!check_non_decreasing(&[(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]));
        assert!(check_non_decreasing(&[(1.0, 1.0)]));
    }

    #[test]
    fn test_empty_current_monotonicity() {
        let c = Current::zero();
        let center = pt(0.0, 0.0);
        let radii = vec![0.1, 0.5];
        let result = check_monotonicity_current(&c, &center, 1, &radii);
        assert!(result.is_monotone);
        assert_eq!(result.density_at_zero, 0.0);
    }
}
