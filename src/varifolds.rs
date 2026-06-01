//! Varifolds — unoriented surfaces for agent boundaries.
//!
//! A varifold is a measure on the Grassmann bundle (space × Grassmannian).
//! Unlike currents, varifolds don't require orientation. This is useful for
//! agent boundaries that may not have a consistent orientation.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::hausdorff::Point;
use crate::currents::Simplex;

/// A varifold: measure on the space of (point, tangent_plane) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Varifold {
    /// Weighted (point, tangent_space) pairs.
    pub elements: Vec<VarifoldElement>,
}

/// A single element of a varifold: a point with a tangent plane and a weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarifoldElement {
    /// Position in ambient space.
    pub point: Point,
    /// Tangent space dimension.
    pub tangent_dimension: usize,
    /// Basis vectors of the tangent space (orthonormal).
    pub tangent_basis: Vec<DVector<f64>>,
    /// Weight (measure).
    pub weight: f64,
}

impl Varifold {
    /// Create an empty varifold.
    pub fn zero() -> Self {
        Self { elements: vec![] }
    }

    /// Total mass (sum of weights).
    pub fn mass(&self) -> f64 {
        self.elements.iter().map(|e| e.weight).sum()
    }

    /// Number of support points.
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Add an element.
    pub fn add(&mut self, element: VarifoldElement) {
        self.elements.push(element);
    }

    /// Create a varifold from a set of unoriented simplices.
    pub fn from_simplices(simplices: &[Simplex]) -> Self {
        let mut elements = Vec::new();
        for simplex in simplices {
            let centroid = simplex.centroid();
            let k = simplex.dimension();

            // Compute tangent space from simplex edges
            let mut basis = Vec::new();
            let v0 = &simplex.vertices[0];
            for i in 1..simplex.vertices.len() {
                let edge = &simplex.vertices[i] - v0;
                let norm = edge.norm();
                if norm > 1e-10 {
                    basis.push(edge / norm);
                }
            }

            elements.push(VarifoldElement {
                point: centroid,
                tangent_dimension: k,
                tangent_basis: basis,
                weight: simplex.volume(),
            });
        }

        Self { elements }
    }

    /// First variation of the varifold (generalized mean curvature).
    /// δV(X) = -Σ weight_i * div_{T_i}(X)(p_i)
    /// For a discrete varifold, this gives the "force" pulling toward minimal surface.
    pub fn first_variation(&self, vector_field: &dyn Fn(&Point) -> DVector<f64>) -> f64 {
        let mut total = 0.0;
        for elem in &self.elements {
            let x = vector_field(&elem.point);
            // Approximate divergence in tangent space
            let div = self.approximate_divergence(&elem.point, &elem.tangent_basis, vector_field);
            total -= elem.weight * div;
        }
        total
    }

    /// Compute the generalized mean curvature vector at each support point.
    pub fn mean_curvature(&self) -> Vec<DVector<f64>> {
        self.elements.iter().map(|elem| {
            // For a k-varifold in R^n, the mean curvature is
            // H = δV / dA projected onto normal space
            let n = elem.point.len();
            if elem.tangent_basis.is_empty() {
                return DVector::zeros(n);
            }

            // Approximate: use Laplacian of position weighted by density
            let mut curvature = DVector::zeros(n);
            for other in &self.elements {
                let diff = &other.point - &elem.point;
                let dist = diff.norm();
                if dist > 1e-10 && dist < 1.0 {
                    let weight = other.weight / (dist * dist * dist);
                    curvature += diff * weight;
                }
            }

            // Project onto normal space
            let normal_component = project_to_normal(&curvature, &elem.tangent_basis);
            normal_component
        }).collect()
    }

    /// Compute the density ratio θ^k(V, x, r) = V(B_r(x)) / (ω_k r^k).
    pub fn density_ratio(&self, center: &Point, radius: f64, k: usize) -> f64 {
        let r2 = radius * radius;
        let mass_in_ball: f64 = self.elements.iter()
            .filter(|e| (&e.point - center).norm_squared() <= r2)
            .map(|e| e.weight)
            .sum();

        let omega_k = crate::hausdorff::volume_unit_ball(k as f64);
        let denom = omega_k * radius.powi(k as i32);
        if denom < 1e-15 { 0.0 } else { mass_in_ball / denom }
    }

    /// Check monotonicity: density ratio should be non-decreasing in r.
    pub fn check_monotonicity(&self, center: &Point, k: usize, radii: &[f64]) -> bool {
        let mut prev = 0.0;
        for &r in radii {
            let dr = self.density_ratio(center, r, k);
            if dr < prev - 1e-10 {
                return false;
            }
            prev = dr;
        }
        true
    }

    /// Varifold distance (a metric on the space of varifolds).
    /// Uses the flat metric for measures.
    pub fn distance_to(&self, other: &Varifold) -> f64 {
        // Simple approximation: compare mass and support
        let mass_diff = (self.mass() - other.mass()).abs();

        // Compare support distributions via earth mover's distance approximation
        let support_diff = if self.elements.is_empty() && other.elements.is_empty() {
            0.0
        } else if self.elements.is_empty() {
            other.mass()
        } else if other.elements.is_empty() {
            self.mass()
        } else {
            // Simple: average nearest neighbor distance
            let mut total = 0.0;
            let n = self.elements.len().max(other.elements.len());
            for elem in &self.elements {
                let min_d = other.elements.iter()
                    .map(|e| (&e.point - &elem.point).norm())
                    .fold(f64::INFINITY, f64::min);
                total += min_d * elem.weight;
            }
            total / self.mass().max(1e-15)
        };

        mass_diff + support_diff
    }

    fn approximate_divergence(
        &self,
        point: &Point,
        basis: &[DVector<f64>],
        field: &dyn Fn(&Point) -> DVector<f64>,
    ) -> f64 {
        let h = 1e-6;
        let mut div = 0.0;
        for b in basis {
            let p_plus = point + b * h;
            let p_minus = point - b * h;
            let f_plus = field(&p_plus);
            let f_minus = field(&p_minus);
            div += (f_plus - f_minus).dot(b) / (2.0 * h);
        }
        div
    }
}

/// Project a vector onto the normal space of a tangent plane.
fn project_to_normal(v: &DVector<f64>, tangent_basis: &[DVector<f64>]) -> DVector<f64> {
    let mut proj = v.clone();
    for b in tangent_basis {
        let component = v.dot(b);
        proj -= b * component;
    }
    proj
}

/// Create a varifold from a point cloud with tangent estimation.
pub fn varifold_from_pointcloud(
    points: &[Point],
    k: usize,
    neighborhood_radius: f64,
) -> Varifold {
    let mut varifold = Varifold::zero();

    for i in 0..points.len() {
        let neighbors: Vec<usize> = points.iter().enumerate()
            .filter(|(j, p)| *j != i && (*p - &points[i]).norm_squared() <= neighborhood_radius * neighborhood_radius)
            .map(|(j, _)| j)
            .collect();

        let tangent_basis = if neighbors.len() >= k {
            estimate_tangent_space(points, i, &neighbors, k)
        } else {
            Vec::new()
        };

        varifold.add(VarifoldElement {
            point: points[i].clone(),
            tangent_dimension: k,
            tangent_basis,
            weight: 1.0 / points.len() as f64,
        });
    }

    varifold
}

fn estimate_tangent_space(
    points: &[Point],
    center_idx: usize,
    neighbors: &[usize],
    k: usize,
) -> Vec<DVector<f64>> {
    if neighbors.is_empty() {
        return Vec::new();
    }

    let dim = points[0].len();
    let n = neighbors.len() as f64;

    // Centroid
    let mut centroid = DVector::zeros(dim);
    for &idx in neighbors {
        centroid += &points[idx];
    }
    centroid /= n;

    // Covariance
    let mut cov = vec![vec![0.0; dim]; dim];
    for &idx in neighbors {
        let diff = &points[idx] - &centroid;
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] += diff[i] * diff[j];
            }
        }
    }

    // Power iteration for top k eigenvectors
    let mut basis = Vec::new();
    let mut deflated = cov.clone();

    for _ in 0..k {
        let mut v = vec![0.0; dim];
        v[0] = 1.0;

        for _ in 0..100 {
            let mut new_v = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    new_v[i] += deflated[i][j] * v[j];
                }
            }
            let norm = (new_v.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if norm < 1e-15 { break; }
            for x in &mut new_v { *x /= norm; }
            v = new_v;
        }

        let eigenvalue: f64 = v.iter().enumerate()
            .map(|(i, vi)| (0..dim).map(|j| vi * deflated[i][j] * v[j]).sum::<f64>())
            .sum();

        if eigenvalue > 1e-10 {
            basis.push(DVector::from_vec(v.clone()));
            // Deflate
            for i in 0..dim {
                for j in 0..dim {
                    deflated[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
    }

    basis
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
    fn test_varifold_zero() {
        let v = Varifold::zero();
        assert_eq!(v.mass(), 0.0);
        assert_eq!(v.size(), 0);
    }

    #[test]
    fn test_varifold_from_simplices() {
        use crate::currents::Simplex;
        let simplices = vec![
            Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0),
            Simplex::new(vec![pt(1.0, 0.0), pt(1.0, 1.0)], 1.0),
        ];
        let v = Varifold::from_simplices(&simplices);
        assert_eq!(v.size(), 2);
        assert!(v.mass() > 0.0);
    }

    #[test]
    fn test_varifold_mass() {
        let mut v = Varifold::zero();
        v.add(VarifoldElement {
            point: pt(0.0, 0.0),
            tangent_dimension: 1,
            tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
            weight: 2.0,
        });
        v.add(VarifoldElement {
            point: pt(1.0, 0.0),
            tangent_dimension: 1,
            tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
            weight: 3.0,
        });
        assert!((v.mass() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_ratio() {
        let mut v = Varifold::zero();
        for i in 0..10 {
            v.add(VarifoldElement {
                point: pt(i as f64, 0.0),
                tangent_dimension: 1,
                tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
                weight: 1.0,
            });
        }
        let dr = v.density_ratio(&pt(5.0, 0.0), 2.0, 1);
        assert!(dr > 0.0);
    }

    #[test]
    fn test_varifold_from_pointcloud() {
        let pts: Vec<Point> = (0..20)
            .map(|i| pt(i as f64 / 20.0, 0.0))
            .collect();
        let v = varifold_from_pointcloud(&pts, 1, 0.2);
        assert_eq!(v.size(), 20);
        assert!(v.mass() > 0.0);
    }

    #[test]
    fn test_mean_curvature() {
        let mut v = Varifold::zero();
        // Flat line — curvature should be small
        for i in 0..10 {
            v.add(VarifoldElement {
                point: pt(i as f64, 0.0),
                tangent_dimension: 1,
                tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
                weight: 1.0,
            });
        }
        let curvatures = v.mean_curvature();
        assert_eq!(curvatures.len(), 10);
    }

    #[test]
    fn test_varifold_distance() {
        let mut v1 = Varifold::zero();
        v1.add(VarifoldElement {
            point: pt(0.0, 0.0),
            tangent_dimension: 1,
            tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
            weight: 1.0,
        });

        let v2 = v1.clone();
        assert!(v1.distance_to(&v2) < 1e-10);

        let mut v3 = Varifold::zero();
        v3.add(VarifoldElement {
            point: pt(10.0, 10.0),
            tangent_dimension: 1,
            tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
            weight: 1.0,
        });
        assert!(v1.distance_to(&v3) > 1.0);
    }

    #[test]
    fn test_project_to_normal() {
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let basis = vec![DVector::from_vec(vec![1.0, 0.0])];
        let normal = project_to_normal(&v, &basis);
        assert!((normal[0]).abs() < 1e-10);
        assert!((normal[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_first_variation() {
        let mut v = Varifold::zero();
        v.add(VarifoldElement {
            point: pt(0.0, 0.0),
            tangent_dimension: 1,
            tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
            weight: 1.0,
        });
        let fv = v.first_variation(&|_p| DVector::from_vec(vec![1.0, 0.0]));
        // Constant field has zero divergence
        assert!(fv.abs() < 1e-4);
    }

    #[test]
    fn test_varifold_serialization() {
        let mut v = Varifold::zero();
        v.add(VarifoldElement {
            point: pt(1.0, 2.0),
            tangent_dimension: 1,
            tangent_basis: vec![DVector::from_vec(vec![1.0, 0.0])],
            weight: 3.0,
        });
        let json = serde_json::to_string(&v).unwrap();
        let v2: Varifold = serde_json::from_str(&json).unwrap();
        assert_eq!(v2.size(), 1);
        assert!((v2.mass() - 3.0).abs() < 1e-10);
    }
}
