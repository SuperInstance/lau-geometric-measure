//! Plateau problem — find minimal surface spanning a given boundary.
//!
//! The Plateau problem: given a (k-1)-dimensional boundary, find the k-dimensional
//! surface of minimal area that spans it. This is the geometric measure theory
//! analog of "what shape does a soap film take?"

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::currents::{Current, Simplex};
use crate::varifolds::Varifold;
use crate::hausdorff::Point;

/// Result of solving the Plateau problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauSolution {
    /// The minimal surface as a current.
    pub surface: Current,
    /// The area of the minimal surface.
    pub area: f64,
    /// The boundary current.
    pub boundary: Current,
    /// Number of iterations used.
    pub iterations: usize,
    /// Whether the solution converged.
    pub converged: bool,
}

/// Solve the discrete Plateau problem for a given boundary.
///
/// Given a boundary current (k-1 dimensional), finds a k-dimensional current
/// with minimal mass having that boundary.
///
/// Uses an iterative relaxation / mean curvature flow approach.
pub fn solve_plateau(
    boundary: &Current,
    initial_fill: Option<&Current>,
    max_iterations: usize,
    tolerance: f64,
) -> PlateauSolution {
    let k = boundary.simplices.first().map(|s| s.dimension() + 1).unwrap_or(0);
    if k == 0 {
        return PlateauSolution {
            surface: Current::zero(),
            area: 0.0,
            boundary: boundary.clone(),
            iterations: 0,
            converged: true,
        };
    }

    // Start with initial fill or a naive fill
    let mut surface = initial_fill.cloned().unwrap_or_else(|| naive_fill(boundary));

    let mut prev_mass = surface.mass();
    let mut converged = false;
    let mut iterations = 0;

    for i in 0..max_iterations {
        iterations = i + 1;
        surface = relax_surface(&surface);
        let new_mass = surface.mass();

        if (prev_mass - new_mass).abs() < tolerance {
            converged = true;
            break;
        }
        prev_mass = new_mass;
    }

    PlateauSolution {
        surface: surface.clone(),
        area: surface.mass(),
        boundary: surface.boundary(),
        iterations,
        converged,
    }
}

/// Solve Plateau problem using varifold approach (unoriented).
pub fn solve_plateau_varifold(
    boundary_simplices: &[Simplex],
    interior_points: &[Point],
    max_iterations: usize,
) -> PlateauSolution {
    if interior_points.is_empty() {
        let c = Current { simplices: boundary_simplices.to_vec() };
        return PlateauSolution {
            surface: c.clone(),
            area: c.mass(),
            boundary: c.boundary(),
            iterations: 0,
            converged: true,
        };
    }

    // Build fan triangulation from boundary to interior
    let mut surface = Current::zero();

    // Triangulate: connect each boundary edge to nearest interior point
    for simplex in boundary_simplices {
        if simplex.dimension() == 1 {
            let centroid = simplex.centroid();
            let nearest = find_nearest(&centroid, interior_points);
            surface.add(Simplex::new(
                vec![simplex.vertices[0].clone(), simplex.vertices[1].clone(), interior_points[nearest].clone()],
                simplex.orientation,
            ));
        }
    }

    let area = surface.mass();
    PlateauSolution {
        surface,
        area,
        boundary: boundary_simplices.first().map(|_| Current::zero()).unwrap_or_else(Current::zero),
        iterations: max_iterations,
        converged: false,
    }
}

/// Compare area of a candidate surface to the isoperimetric lower bound.
pub fn compare_to_isoperimetric_bound(surface: &Current, k: usize) -> f64 {
    let area = surface.mass();
    let bdry = surface.boundary();
    let bdry_length = bdry.mass();

    // Isoperimetric: area ≥ C * bdry_length^(k/(k-1))
    if k <= 1 || bdry_length < 1e-10 {
        return f64::INFINITY;
    }

    let c = isoperimetric_constant(k);
    let lower_bound = c * bdry_length.powf(k as f64 / (k - 1) as f64);

    if lower_bound < 1e-15 {
        return f64::INFINITY;
    }

    area / lower_bound
}

/// Isoperimetric constant for R^k.
/// For k=2, C = 1/(4π). For k=3, C = (36π)^(1/3).
fn isoperimetric_constant(k: usize) -> f64 {
    match k {
        1 => 0.5,
        2 => 1.0 / (4.0 * std::f64::consts::PI),
        3 => (36.0 * std::f64::consts::PI).powf(1.0 / 3.0),
        _ => {
            // General: using Federer's formula
            let n = k as f64;
            let omega_k = crate::hausdorff::volume_unit_ball(n);
            let omega_k1 = crate::hausdorff::volume_unit_ball(n - 1.0);
            n.powf(n / (n - 1.0)) * omega_k1.powf(n / (n - 1.0)) / omega_k
        }
    }
}

/// Minimal surface area for a circle of given radius.
pub fn minimal_disk_area(radius: f64) -> f64 {
    std::f64::consts::PI * radius * radius
}

/// Minimal surface area for a given boundary curve (using area formula).
pub fn estimate_minimal_area(boundary: &Current) -> f64 {
    // For a 1-dimensional boundary in R^2, estimate enclosed area
    // using the shoelace formula
    let points: Vec<Point> = boundary.simplices.iter()
        .map(|s| s.vertices[0].clone())
        .collect();

    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }
    area.abs() / 2.0
}

// --- Internal helpers ---

fn naive_fill(boundary: &Current) -> Current {
    if boundary.simplices.is_empty() {
        return Current::zero();
    }

    let dim = boundary.simplices[0].vertices[0].len();
    let mut surface = Current::zero();

    // For 1D boundary (edges), try to create 2D triangles via fan
    let mut all_points: Vec<Point> = Vec::new();
    for s in &boundary.simplices {
        for v in &s.vertices {
            all_points.push(v.clone());
        }
    }

    if all_points.len() < 3 {
        return Current::zero();
    }

    // Centroid as fan center
    let n = all_points.len() as f64;
    let mut centroid = DVector::zeros(dim);
    for p in &all_points {
        centroid += p;
    }
    centroid /= n;

    // Connect each boundary edge to centroid
    for s in &boundary.simplices {
        if s.vertices.len() == 2 {
            surface.add(Simplex::new(
                vec![s.vertices[0].clone(), s.vertices[1].clone(), centroid.clone()],
                s.orientation,
            ));
        }
    }

    surface
}

fn relax_surface(surface: &Current) -> Current {
    // Move each interior vertex toward the centroid of its neighbors
    // (Laplacian smoothing — discrete mean curvature flow)

    // Collect all unique vertices
    let mut vertex_positions: Vec<Point> = Vec::new();
    for s in &surface.simplices {
        for v in &s.vertices {
            if !vertex_positions.iter().any(|p| (p - v).norm_squared() < 1e-15) {
                vertex_positions.push(v.clone());
            }
        }
    }

    // Find boundary vertices (vertices in boundary simplices)
    let boundary = surface.boundary();
    let boundary_verts: Vec<Point> = boundary.simplices.iter()
        .flat_map(|s| s.vertices.clone())
        .collect();

    // Build adjacency
    let mut new_positions = vertex_positions.clone();
    let alpha = 0.3; // Relaxation parameter

    for (idx, vertex) in vertex_positions.iter().enumerate() {
        // Skip boundary vertices
        if boundary_verts.iter().any(|bv| (bv - vertex).norm_squared() < 1e-10) {
            continue;
        }

        // Find neighbors via shared simplices
        let mut neighbors = Vec::new();
        for s in &surface.simplices {
            if s.vertices.iter().any(|v| (v - vertex).norm_squared() < 1e-10) {
                for v in &s.vertices {
                    if (v - vertex).norm_squared() > 1e-10 {
                        neighbors.push(v.clone());
                    }
                }
            }
        }

        if neighbors.is_empty() {
            continue;
        }

        let mut centroid = DVector::zeros(vertex.len());
        for n in &neighbors {
            centroid += n;
        }
        centroid /= neighbors.len() as f64;

        new_positions[idx] = vertex + (centroid - vertex) * alpha;
    }

    // Rebuild surface with new positions
    let mut new_surface = Current::zero();
    for s in &surface.simplices {
        let new_verts: Vec<Point> = s.vertices.iter().map(|v| {
            let idx = vertex_positions.iter().position(|p| (p - v).norm_squared() < 1e-10).unwrap();
            new_positions[idx].clone()
        }).collect();
        new_surface.add(Simplex::new(new_verts, s.orientation));
    }

    new_surface
}

fn find_nearest(point: &Point, candidates: &[Point]) -> usize {
    candidates.iter()
        .enumerate()
        .map(|(i, p)| (i, (p - point).norm_squared()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point {
        DVector::from_vec(vec![x, y])
    }

    #[test]
    fn test_plateau_triangle_boundary() {
        // Boundary of a triangle
        let mut boundary = Current::zero();
        boundary.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(1.0, 0.0), pt(0.5, 1.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(0.5, 1.0), pt(0.0, 0.0)], 1.0));

        let solution = solve_plateau(&boundary, None, 100, 1e-6);
        assert!(solution.area > 0.0);
        assert!(solution.iterations <= 100);
    }

    #[test]
    fn test_plateau_empty_boundary() {
        let boundary = Current::zero();
        let solution = solve_plateau(&boundary, None, 100, 1e-6);
        assert_eq!(solution.area, 0.0);
        assert!(solution.converged);
    }

    #[test]
    fn test_plateau_with_initial_fill() {
        let mut boundary = Current::zero();
        boundary.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(1.0, 0.0), pt(0.0, 1.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(0.0, 1.0), pt(0.0, 0.0)], 1.0));

        let initial = Current::from_simplex(Simplex::new(
            vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)],
            1.0,
        ));

        let solution = solve_plateau(&boundary, Some(&initial), 100, 1e-6);
        assert!(solution.area > 0.0);
    }

    #[test]
    fn test_naive_fill() {
        let mut boundary = Current::zero();
        boundary.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(1.0, 0.0), pt(0.0, 1.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(0.0, 1.0), pt(0.0, 0.0)], 1.0));

        let fill = naive_fill(&boundary);
        assert!(fill.size() > 0);
        assert!(fill.mass() > 0.0);
    }

    #[test]
    fn test_minimal_disk_area() {
        let area = minimal_disk_area(1.0);
        assert!((area - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_minimal_area() {
        let mut boundary = Current::zero();
        boundary.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(1.0, 0.0), pt(1.0, 1.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(1.0, 1.0), pt(0.0, 1.0)], 1.0));
        boundary.add(Simplex::new(vec![pt(0.0, 1.0), pt(0.0, 0.0)], 1.0));

        let area = estimate_minimal_area(&boundary);
        assert!((area - 1.0).abs() < 0.5); // Rough estimate
    }

    #[test]
    fn test_isoperimetric_constant() {
        let c2 = isoperimetric_constant(2);
        assert!(c2 > 0.0);
        let c3 = isoperimetric_constant(3);
        assert!(c3 > 0.0);
    }

    #[test]
    fn test_compare_to_isoperimetric_bound() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let surface = Current::from_simplex(tri);
        let ratio = compare_to_isoperimetric_bound(&surface, 2);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_relax_surface() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let surface = Current::from_simplex(tri);
        let relaxed = relax_surface(&surface);
        assert!(relaxed.mass() > 0.0);
    }

    #[test]
    fn test_plateau_varifold() {
        let boundary = vec![
            Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0),
            Simplex::new(vec![pt(1.0, 0.0), pt(0.5, 1.0)], 1.0),
        ];
        let interior = vec![pt(0.5, 0.3)];
        let solution = solve_plateau_varifold(&boundary, &interior, 10);
        assert!(solution.area > 0.0);
    }

    #[test]
    fn test_solution_serialization() {
        let solution = PlateauSolution {
            surface: Current::zero(),
            area: 0.0,
            boundary: Current::zero(),
            iterations: 0,
            converged: true,
        };
        let json = serde_json::to_string(&solution).unwrap();
        let s2: PlateauSolution = serde_json::from_str(&json).unwrap();
        assert!(s2.converged);
        assert_eq!(s2.area, 0.0);
    }
}
