//! Agent state space dimension measurement.
//!
//! Applies geometric measure theory tools to measure the effective dimension
//! and structure of agent state spaces, which may have fractal or non-integer
//! dimension.

use serde::{Serialize, Deserialize};
use crate::hausdorff::{Point, hausdorff_dimension_auto, hausdorff_measure};
use crate::rectifiability::{test_rectifiability, Rectifiability};
use crate::currents::Current;
use crate::varifolds::Varifold;

/// Description of an agent's state space geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateSpaceGeometry {
    /// Estimated Hausdorff dimension of the state space.
    pub hausdorff_dimension: f64,
    /// Confidence of the dimension estimate.
    pub dimension_confidence: f64,
    /// Whether the state space is rectifiable.
    pub is_rectifiable: bool,
    /// Estimated dimension of rectifiable part.
    pub rectifiable_dimension: Option<usize>,
    /// Total mass of the state space (Hausdorff measure).
    pub total_mass: f64,
    /// Number of sample points used.
    pub sample_count: usize,
}

/// Measure the geometry of an agent's state space from sample points.
pub fn measure_agent_state_space(
    state_points: &[Point],
    epsilon: f64,
) -> AgentStateSpaceGeometry {
    if state_points.is_empty() {
        return AgentStateSpaceGeometry {
            hausdorff_dimension: 0.0,
            dimension_confidence: 0.0,
            is_rectifiable: true,
            rectifiable_dimension: Some(0),
            total_mass: 0.0,
            sample_count: 0,
        };
    }

    // Estimate Hausdorff dimension
    let dim_result = hausdorff_dimension_auto(state_points);
    let _dim_floor = dim_result.dimension.floor() as usize;
    let dim_rounded = dim_result.dimension.round() as usize;

    // Check rectifiability
    let rect_result = test_rectifiability(state_points, dim_rounded.max(1), epsilon * 10.0, 20);
    let is_rectifiable = matches!(rect_result.classification, Rectifiability::Rectifiable { .. } | Rectifiability::Partial { .. });

    // Compute Hausdorff measure at the estimated dimension
    let mass = hausdorff_measure(state_points, dim_result.dimension, epsilon).measure;

    AgentStateSpaceGeometry {
        hausdorff_dimension: dim_result.dimension,
        dimension_confidence: dim_result.confidence,
        is_rectifiable,
        rectifiable_dimension: if is_rectifiable { Some(dim_rounded) } else { None },
        total_mass: mass,
        sample_count: state_points.len(),
    }
}

/// Detect regime changes in agent behavior from state trajectory.
///
/// Returns indices where the local dimension changes significantly.
pub fn detect_regime_changes(
    trajectory: &[Point],
    window_size: usize,
    threshold: f64,
) -> Vec<usize> {
    if trajectory.len() < window_size * 2 {
        return vec![];
    }

    let mut dimensions: Vec<(usize, f64)> = Vec::new();
    for start in (0..trajectory.len().saturating_sub(window_size)).step_by(window_size / 2) {
        let end = (start + window_size).min(trajectory.len());
        let window = &trajectory[start..end];
        let dim = hausdorff_dimension_auto(window);
        dimensions.push((start, dim.dimension));
    }

    let mut changes = Vec::new();
    for i in 1..dimensions.len() {
        let diff = (dimensions[i].1 - dimensions[i - 1].1).abs();
        if diff > threshold {
            changes.push(dimensions[i].0);
        }
    }

    changes
}

/// Compute the effective exploration coverage of an agent.
///
/// Returns the fraction of the state space that has been explored,
/// estimated via covering numbers.
pub fn exploration_coverage(
    explored_points: &[Point],
    total_state_points: &[Point],
    epsilon: f64,
) -> f64 {
    if total_state_points.is_empty() {
        return 0.0;
    }

    let eps_sq = epsilon * epsilon;
    let mut covered = vec![false; total_state_points.len()];

    for ep in explored_points {
        for (i, tp) in total_state_points.iter().enumerate() {
            if !covered[i] && (ep - tp).norm_squared() <= eps_sq {
                covered[i] = true;
            }
        }
    }

    covered.iter().filter(|&&c| c).count() as f64 / total_state_points.len() as f64
}

/// Agent state space as a current (for boundary analysis).
pub fn state_space_as_current(points: &[Point]) -> Current {
    crate::currents::triangulate_1d(points)
}

/// Agent state space as a varifold (for curvature analysis).
pub fn state_space_as_varifold(points: &[Point], k: usize, radius: f64) -> Varifold {
    crate::varifolds::varifold_from_pointcloud(points, k, radius)
}

/// Compare two agent state spaces.
pub fn compare_state_spaces(
    points_a: &[Point],
    points_b: &[Point],
    epsilon: f64,
) -> StateSpaceComparison {
    let geom_a = measure_agent_state_space(points_a, epsilon);
    let geom_b = measure_agent_state_space(points_b, epsilon);

    let dim_diff = (geom_a.hausdorff_dimension - geom_b.hausdorff_dimension).abs();
    let mass_ratio = if geom_b.total_mass > 1e-10 {
        geom_a.total_mass / geom_b.total_mass
    } else if geom_a.total_mass > 1e-10 {
        f64::INFINITY
    } else {
        1.0
    };

    StateSpaceComparison {
        dimension_difference: dim_diff,
        mass_ratio,
        same_rectifiability: geom_a.is_rectifiable == geom_b.is_rectifiable,
        geometry_a: geom_a,
        geometry_b: geom_b,
    }
}

/// Result of comparing two state spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceComparison {
    pub dimension_difference: f64,
    pub mass_ratio: f64,
    pub same_rectifiability: bool,
    pub geometry_a: AgentStateSpaceGeometry,
    pub geometry_b: AgentStateSpaceGeometry,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn pt(x: f64, y: f64) -> Point {
        DVector::from_vec(vec![x, y])
    }

    #[test]
    fn test_measure_line_state_space() {
        let pts: Vec<Point> = (0..50)
            .map(|i| pt(i as f64 / 50.0, 0.0))
            .collect();
        let geom = measure_agent_state_space(&pts, 0.01);
        assert!((geom.hausdorff_dimension - 1.0).abs() < 0.5);
        assert!(geom.sample_count == 50);
    }

    #[test]
    fn test_measure_empty_state_space() {
        let pts: Vec<Point> = vec![];
        let geom = measure_agent_state_space(&pts, 0.1);
        assert_eq!(geom.hausdorff_dimension, 0.0);
        assert_eq!(geom.sample_count, 0);
    }

    #[test]
    fn test_detect_regime_changes() {
        // First half: line, second half: plane
        let mut pts: Vec<Point> = (0..30)
            .map(|i| pt(i as f64 / 30.0, 0.0))
            .collect();
        pts.extend((0..30).flat_map(|i| {
            (0..3).map(move |j| pt(i as f64 / 30.0, j as f64))
        }));
        let changes = detect_regime_changes(&pts, 10, 0.3);
        // Should detect change around the boundary
        assert!(!changes.is_empty() || pts.len() < 20);
    }

    #[test]
    fn test_exploration_coverage() {
        let total: Vec<Point> = (0..10)
            .map(|i| pt(i as f64, 0.0))
            .collect();
        let explored: Vec<Point> = (0..5)
            .map(|i| pt(i as f64, 0.0))
            .collect();
        let cov = exploration_coverage(&explored, &total, 0.5);
        assert!(cov > 0.0);
        assert!(cov <= 1.0);
    }

    #[test]
    fn test_exploration_full_coverage() {
        let pts: Vec<Point> = (0..5)
            .map(|i| pt(i as f64, 0.0))
            .collect();
        let cov = exploration_coverage(&pts, &pts, 0.5);
        assert!((cov - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_space_as_current() {
        let pts: Vec<Point> = (0..10)
            .map(|i| pt(i as f64, 0.0))
            .collect();
        let c = state_space_as_current(&pts);
        assert!(c.size() > 0);
    }

    #[test]
    fn test_state_space_as_varifold() {
        let pts: Vec<Point> = (0..10)
            .map(|i| pt(i as f64, 0.0))
            .collect();
        let v = state_space_as_varifold(&pts, 1, 0.5);
        assert!(v.size() > 0);
    }

    #[test]
    fn test_compare_state_spaces() {
        let line: Vec<Point> = (0..20)
            .map(|i| pt(i as f64 / 20.0, 0.0))
            .collect();
        let plane: Vec<Point> = (0..5)
            .flat_map(|i| (0..5).map(move |j| pt(i as f64, j as f64)))
            .collect();
        let comp = compare_state_spaces(&line, &plane, 0.1);
        assert!(comp.dimension_difference >= 0.0);
    }

    #[test]
    fn test_exploration_coverage_empty() {
        let pts: Vec<Point> = vec![pt(0.0, 0.0)];
        let cov = exploration_coverage(&pts, &[], 0.5);
        assert_eq!(cov, 0.0);
    }

    #[test]
    fn test_geometry_serialization() {
        let geom = AgentStateSpaceGeometry {
            hausdorff_dimension: 1.5,
            dimension_confidence: 0.9,
            is_rectifiable: true,
            rectifiable_dimension: Some(2),
            total_mass: 3.14,
            sample_count: 100,
        };
        let json = serde_json::to_string(&geom).unwrap();
        let g2: AgentStateSpaceGeometry = serde_json::from_str(&json).unwrap();
        assert!((g2.hausdorff_dimension - 1.5).abs() < 1e-10);
        assert_eq!(g2.sample_count, 100);
    }

    #[test]
    fn test_detect_regime_changes_small() {
        let pts: Vec<Point> = vec![pt(0.0, 0.0), pt(1.0, 0.0)];
        let changes = detect_regime_changes(&pts, 10, 0.5);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_compare_identical_spaces() {
        let pts: Vec<Point> = (0..20)
            .map(|i| pt(i as f64 / 20.0, 0.0))
            .collect();
        let comp = compare_state_spaces(&pts, &pts, 0.1);
        assert!((comp.dimension_difference).abs() < 1e-10);
    }
}
