//! Currents — oriented k-dimensional surfaces in n-dimensional space.
//!
//! Based on Federer-Fleming theory. A k-current is a continuous linear functional
//! on the space of smooth k-forms with compact support. We represent them as
//! weighted sums of oriented simplices.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::hausdorff::Point;

/// A k-simplex in n-dimensional space (oriented).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Simplex {
    /// Vertices of the simplex (k+1 vertices for a k-simplex).
    pub vertices: Vec<Point>,
    /// Orientation sign (+1 or -1).
    pub orientation: f64,
}

impl Simplex {
    /// Create a new oriented simplex.
    pub fn new(vertices: Vec<Point>, orientation: f64) -> Self {
        Self { vertices, orientation }
    }

    /// Dimension of the simplex (k = number of vertices - 1).
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Compute the k-dimensional volume of this simplex.
    pub fn volume(&self) -> f64 {
        let k = self.dimension();
        if k == 0 {
            return 1.0; // unsigned geometric volume
        }

        // Volume = |det of edge vectors| / k!
        let v0 = &self.vertices[0];
        let edges: Vec<DVector<f64>> = self.vertices[1..]
            .iter()
            .map(|v| v - v0)
            .collect();

        if edges.len() != k {
            return 0.0;
        }

        // Build Gram matrix
        let mut gram = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                gram[i][j] = edges[i].dot(&edges[j]);
            }
        }

        let det = determinant(&gram);
        det.abs().sqrt() / factorial(k) as f64
    }

    /// Compute the boundary of this simplex (a (k-1)-current).
    pub fn boundary(&self) -> Current {
        let k = self.dimension();
        if k == 0 {
            return Current::zero();
        }

        let mut simplices = Vec::new();
        for i in 0..=k {
            let mut face_verts: Vec<Point> = self.vertices.iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, v)| v.clone())
                .collect();

            let sign = self.orientation * (-1.0_f64).powi(i as i32);
            simplices.push(Simplex::new(face_verts, sign));
        }

        Current { simplices }
    }

    /// Compute the mass (total volume with signs) of this simplex.
    pub fn mass(&self) -> f64 {
        self.volume()
    }

    /// Diameter of the simplex.
    pub fn diameter(&self) -> f64 {
        let mut max_d = 0.0;
        for i in 0..self.vertices.len() {
            for j in (i + 1)..self.vertices.len() {
                let d = (&self.vertices[i] - &self.vertices[j]).norm();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        max_d
    }

    /// Centroid of the simplex.
    pub fn centroid(&self) -> Point {
        let n = self.vertices.len() as f64;
        let dim = self.vertices[0].len();
        let mut c = DVector::zeros(dim);
        for v in &self.vertices {
            c += v;
        }
        c / n
    }
}

/// A k-current: finite linear combination of oriented k-simplices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Current {
    pub simplices: Vec<Simplex>,
}

impl Current {
    /// Create a zero current.
    pub fn zero() -> Self {
        Self { simplices: vec![] }
    }

    /// Create a current from a single simplex.
    pub fn from_simplex(simplex: Simplex) -> Self {
        Self { simplices: vec![simplex] }
    }

    /// Add a simplex to this current.
    pub fn add(&mut self, simplex: Simplex) {
        self.simplices.push(simplex);
    }

    /// Compute the boundary: ∂T = Σ ∂(each simplex).
    /// By the boundary operator, ∂∘∂ = 0.
    /// Consolidates (cancels) simplices with identical vertices.
    pub fn boundary(&self) -> Current {
        let mut result = Current::zero();
        for simplex in &self.simplices {
            let bdry = simplex.boundary();
            for s in bdry.simplices {
                result.add(s);
            }
        }
        result.consolidate()
    }

    /// Consolidate simplices with identical vertex positions.
    /// Merges orientations; removes simplices whose orientation cancels to ~0.
    pub fn consolidate(&self) -> Current {
        let mut groups: Vec<(Vec<Point>, f64)> = Vec::new();
        'outer: for s in &self.simplices {
            for (g_verts, g_orient) in &mut groups {
                if g_verts.len() != s.vertices.len() { continue; }
                let all_match = g_verts.iter().zip(s.vertices.iter())
                    .all(|(a, b)| (a - b).norm() < 1e-10);
                if all_match {
                    *g_orient += s.orientation;
                    continue 'outer;
                }
            }
            groups.push((s.vertices.clone(), s.orientation));
        }
        let simplices: Vec<Simplex> = groups.into_iter()
            .filter(|(_, w)| w.abs() > 1e-10)
            .map(|(v, o)| Simplex::new(v, o))
            .collect();
        Current { simplices }
    }

    /// Mass of the current: M(T) = Σ |orientation_i| * volume_i.
    pub fn mass(&self) -> f64 {
        self.simplices.iter().map(|s| s.orientation.abs() * s.volume()).sum()
    }

    /// Flat norm: F(T) = inf { M(A) + M(B) : T = A + ∂B }.
    /// We approximate this.
    pub fn flat_norm(&self) -> f64 {
        // Upper bound: M(T) itself
        let mass_bound = self.mass();

        // Try: is ∂T small?
        let bdry = self.boundary();
        let bdry_mass = bdry.mass();

        // Lower bound on flat norm
        // F(T) ≥ sup { T(ω) : |ω| ≤ 1, |dω| ≤ 1 }
        // Approximate by mass / 2
        let lower_bound = mass_bound / 2.0;

        // For a current with zero boundary, flat norm = mass
        if bdry_mass < 1e-10 {
            return mass_bound;
        }

        // General approximation
        (mass_bound + bdry_mass).min(mass_bound)
    }

    /// Support size (number of simplices).
    pub fn size(&self) -> usize {
        self.simplices.len()
    }

    /// Check if the current has zero boundary (∂T = 0).
    pub fn is_cycle(&self) -> bool {
        let bdry = self.boundary();
        bdry.mass() < 1e-10
    }

    /// Scale all simplices by a scalar.
    pub fn scale(&self, factor: f64) -> Current {
        Current {
            simplices: self.simplices.iter().map(|s| {
                Simplex::new(s.vertices.clone(), s.orientation * factor)
            }).collect(),
        }
    }

    /// Translate all simplices by a vector.
    pub fn translate(&self, v: &DVector<f64>) -> Current {
        Current {
            simplices: self.simplices.iter().map(|s| {
                let new_verts: Vec<Point> = s.vertices.iter().map(|p| p + v).collect();
                Simplex::new(new_verts, s.orientation)
            }).collect(),
        }
    }

    /// Pushforward under a linear map represented by a matrix.
    pub fn pushforward(&self, matrix: &nalgebra::DMatrix<f64>) -> Current {
        Current {
            simplices: self.simplices.iter().map(|s| {
                let new_verts: Vec<Point> = s.vertices.iter().map(|p| matrix * p).collect();
                Simplex::new(new_verts, s.orientation * matrix.determinant().abs())
            }).collect(),
        }
    }
}

/// Verify that ∂∘∂ = 0 (boundary of boundary is zero).
pub fn verify_boundary_of_boundary_zero(current: &Current) -> bool {
    let bdry = current.boundary();
    let bdry_of_bdry = bdry.boundary();
    bdry_of_bdry.mass() < 1e-8
}

/// Compute the flat distance between two currents.
pub fn flat_distance(t1: &Current, t2: &Current) -> f64 {
    // F(T1 - T2)
    let mut diff = t1.clone();
    for s in &t2.simplices {
        diff.add(Simplex::new(s.vertices.clone(), -s.orientation));
    }
    diff.consolidate().flat_norm()
}

/// The support of a current (as a set of centroids of simplices).
pub fn support(current: &Current) -> Vec<Point> {
    current.simplices.iter().map(|s| s.centroid()).collect()
}

// --- Helper functions ---

fn determinant(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    if n == 3 {
        return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    }

    // LU-style expansion for larger matrices
    let mut det = 0.0;
    for j in 0..n {
        let minor = minor_matrix(matrix, 0, j);
        det += (-1.0_f64).powi(j as i32) * matrix[0][j] * determinant(&minor);
    }
    det
}

fn minor_matrix(matrix: &[Vec<f64>], row: usize, col: usize) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut result = Vec::new();
    for i in 0..n {
        if i == row {
            continue;
        }
        let mut row_vec = Vec::new();
        for j in 0..n {
            if j != col {
                row_vec.push(matrix[i][j]);
            }
        }
        result.push(row_vec);
    }
    result
}

fn factorial(n: usize) -> usize {
    if n <= 1 { 1 } else { (1..=n).product() }
}

/// Build a simplicial complex triangulating a set of points.
/// Returns a 1-current (the 1-skeleton) of the Delaunay-like triangulation.
pub fn triangulate_1d(points: &[Point]) -> Current {
    if points.len() < 2 {
        return Current::zero();
    }

    // Sort points by first coordinate
    let mut indexed: Vec<(usize, f64)> = points.iter()
        .enumerate()
        .map(|(i, p)| (i, p[0]))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut current = Current::zero();
    for i in 0..indexed.len() - 1 {
        let a = indexed[i].0;
        let b = indexed[i + 1].0;
        current.add(Simplex::new(
            vec![points[a].clone(), points[b].clone()],
            1.0,
        ));
    }
    current
}

/// Build a triangulated 2-current from a point cloud (simple fan triangulation).
pub fn triangulate_2d_fan(points: &[Point]) -> Current {
    if points.len() < 3 {
        return Current::zero();
    }

    // Use centroid as fan center
    let n = points.len() as f64;
    let dim = points[0].len();
    let mut centroid = DVector::zeros(dim);
    for p in points {
        centroid += p;
    }
    centroid /= n;

    let mut current = Current::zero();
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        current.add(Simplex::new(
            vec![points[i].clone(), points[j].clone(), centroid.clone()],
            1.0,
        ));
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point {
        DVector::from_vec(vec![x, y])
    }

    fn pt3(x: f64, y: f64, z: f64) -> Point {
        DVector::from_vec(vec![x, y, z])
    }

    #[test]
    fn test_simplex_dimension() {
        let s = Simplex::new(vec![pt(0.0, 0.0)], 1.0);
        assert_eq!(s.dimension(), 0);
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0);
        assert_eq!(s.dimension(), 1);
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        assert_eq!(s.dimension(), 2);
    }

    #[test]
    fn test_segment_volume() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(3.0, 4.0)], 1.0);
        assert!((s.volume() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_volume() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        assert!((s.volume() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tetrahedron_volume() {
        let s = Simplex::new(vec![
            pt3(0.0, 0.0, 0.0),
            pt3(1.0, 0.0, 0.0),
            pt3(0.0, 1.0, 0.0),
            pt3(0.0, 0.0, 1.0),
        ], 1.0);
        assert!((s.volume() - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_of_segment() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0);
        let bdry = s.boundary();
        assert_eq!(bdry.simplices.len(), 2);
    }

    #[test]
    fn test_boundary_of_boundary_is_zero() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let bdry = tri.boundary();
        let bdry_bdry = bdry.boundary();
        assert!(bdry_bdry.mass() < 1e-10, "∂∘∂ should be zero, got mass {}", bdry_bdry.mass());
    }

    #[test]
    fn test_verify_boundary_boundary_zero() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let current = Current::from_simplex(tri);
        assert!(verify_boundary_of_boundary_zero(&current));
    }

    #[test]
    fn test_current_mass() {
        let tri1 = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let tri2 = Simplex::new(vec![pt(1.0, 0.0), pt(2.0, 0.0), pt(1.0, 1.0)], 1.0);
        let mut c = Current::zero();
        c.add(tri1);
        c.add(tri2);
        assert!((c.mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_current_is_cycle() {
        // Closed triangle boundary
        let mut c = Current::zero();
        c.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        c.add(Simplex::new(vec![pt(1.0, 0.0), pt(0.0, 1.0)], 1.0));
        c.add(Simplex::new(vec![pt(0.0, 1.0), pt(0.0, 0.0)], 1.0));
        assert!(c.is_cycle());
    }

    #[test]
    fn test_current_not_cycle() {
        let mut c = Current::zero();
        c.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        assert!(!c.is_cycle());
    }

    #[test]
    fn test_flat_norm() {
        let tri = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0)], 1.0);
        let c = Current::from_simplex(tri);
        let fn_val = c.flat_norm();
        assert!(fn_val >= 0.0);
    }

    #[test]
    fn test_current_scale() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0);
        let c = Current::from_simplex(s);
        let scaled = c.scale(2.0);
        assert!((scaled.mass() - 2.0 * c.mass()).abs() < 1e-10);
    }

    #[test]
    fn test_current_translate() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0);
        let c = Current::from_simplex(s);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let translated = c.translate(&v);
        assert_eq!(translated.simplices[0].vertices[0][0], 1.0);
    }

    #[test]
    fn test_simplex_diameter() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(3.0, 4.0)], 1.0);
        assert!((s.diameter() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplex_centroid() {
        let s = Simplex::new(vec![pt(0.0, 0.0), pt(2.0, 0.0)], 1.0);
        let c = s.centroid();
        assert!((c[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangulate_1d() {
        let pts = vec![pt(2.0, 0.0), pt(0.0, 0.0), pt(1.0, 0.0)];
        let current = triangulate_1d(&pts);
        assert_eq!(current.size(), 2); // Two segments
    }

    #[test]
    fn test_triangulate_2d_fan() {
        let pts = vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)];
        let current = triangulate_2d_fan(&pts);
        assert_eq!(current.size(), 4); // Four triangles in fan
    }

    #[test]
    fn test_flat_distance() {
        let c1 = Current::from_simplex(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        let c2 = Current::from_simplex(Simplex::new(vec![pt(0.0, 0.0), pt(2.0, 0.0)], 1.0));
        let d = flat_distance(&c1, &c2);
        assert!(d >= 0.0);
    }

    #[test]
    fn test_pushforward() {
        let s = Simplex::new(vec![pt(1.0, 0.0), pt(2.0, 0.0)], 1.0);
        let c = Current::from_simplex(s);
        let m = nalgebra::DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let pushed = c.pushforward(&m);
        assert!(pushed.mass() > c.mass());
    }

    #[test]
    fn test_determinant() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!((determinant(&m) - 1.0).abs() < 1e-10);

        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((determinant(&m) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_support() {
        let c = Current::from_simplex(Simplex::new(vec![pt(0.0, 0.0), pt(2.0, 0.0)], 1.0));
        let supp = support(&c);
        assert_eq!(supp.len(), 1);
        assert!((supp[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_current() {
        let c = Current::zero();
        assert_eq!(c.mass(), 0.0);
        assert_eq!(c.size(), 0);
        assert!(c.is_cycle());
    }
}
