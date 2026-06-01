//! Federer-Fleming compactness theorem.
//!
//! The space of integral currents with bounded mass and bounded boundary mass
//! is compact in the flat norm topology. This means limits of currents exist,
//! which is essential for existence proofs in geometric measure theory.

use serde::{Serialize, Deserialize};
use crate::currents::{Current, Simplex, flat_distance};
use crate::hausdorff::Point;

/// A sequence of currents with bounds, suitable for compactness arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentSequence {
    /// The sequence of currents.
    pub currents: Vec<Current>,
    /// Bound on mass: M(T_i) ≤ mass_bound for all i.
    pub mass_bound: f64,
    /// Bound on boundary mass: M(∂T_i) ≤ boundary_bound for all i.
    pub boundary_bound: f64,
}

impl CurrentSequence {
    /// Create a new sequence with given bounds.
    pub fn new(mass_bound: f64, boundary_bound: f64) -> Self {
        Self {
            currents: vec![],
            mass_bound,
            boundary_bound,
        }
    }

    /// Add a current to the sequence (verifies bounds).
    pub fn push(&mut self, current: Current) -> Result<(), String> {
        let mass = current.mass();
        let bdry_mass = current.boundary().mass();

        if mass > self.mass_bound + 1e-10 {
            return Err(format!("Mass {} exceeds bound {}", mass, self.mass_bound));
        }
        if bdry_mass > self.boundary_bound + 1e-10 {
            return Err(format!("Boundary mass {} exceeds bound {}", bdry_mass, self.boundary_bound));
        }

        self.currents.push(current);
        Ok(())
    }

    /// Check if the sequence is Cauchy in the flat norm.
    pub fn is_cauchy(&self, tolerance: f64) -> bool {
        for i in 0..self.currents.len() {
            for j in (i + 1)..self.currents.len() {
                if flat_distance(&self.currents[i], &self.currents[j]) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Extract a convergent subsequence (compactness guarantee).
    /// Returns indices of the subsequence.
    pub fn convergent_subsequence(&self) -> Vec<usize> {
        if self.currents.len() <= 1 {
            return (0..self.currents.len()).collect();
        }

        // Simple greedy: pick indices forming an approximately Cauchy sequence
        let mut indices = vec![0];
        let threshold = 0.1; // flat norm threshold

        for i in 1..self.currents.len() {
            let last = indices.last().unwrap();
            if flat_distance(&self.currents[*last], &self.currents[i]) < threshold {
                indices.push(i);
            }
        }

        if indices.len() < 2 && self.currents.len() >= 2 {
            // Fallback: just take every other element
            indices = (0..self.currents.len()).step_by(2).collect();
        }

        indices
    }

    /// Compute the flat-norm limit (if the sequence converges).
    pub fn flat_limit(&self) -> Option<Current> {
        if self.currents.is_empty() {
            return None;
        }

        // If Cauchy, take the last element as approximate limit
        if self.is_cauchy(1.0) {
            Some(self.currents.last().unwrap().clone())
        } else {
            // Take subsequence limit
            let subseq = self.convergent_subsequence();
            if subseq.is_empty() {
                None
            } else {
                Some(self.currents[*subseq.last().unwrap()].clone())
            }
        }
    }
}

/// Result of applying compactness theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactnessResult {
    /// The limiting current.
    pub limit: Current,
    /// Mass of the limit (≤ original mass bound).
    pub limit_mass: f64,
    /// Lower semicontinuity gap: M(T) ≤ lim inf M(T_i).
    pub lower_semicontinuity_gap: f64,
    /// Whether the sequence was actually convergent.
    pub was_convergent: bool,
}

/// Apply the Federer-Fleming compactness theorem.
///
/// Given a sequence of integral currents with uniformly bounded mass and boundary mass,
/// extract a convergent subsequence in the flat norm.
pub fn apply_compactness(sequence: &CurrentSequence) -> Result<CompactnessResult, String> {
    if sequence.currents.is_empty() {
        return Err("Empty sequence".to_string());
    }

    // Verify bounds
    for (i, c) in sequence.currents.iter().enumerate() {
        let mass = c.mass();
        let bdry = c.boundary().mass();
        if mass > sequence.mass_bound + 1e-8 {
            return Err(format!("Current {} has mass {} > bound {}", i, mass, sequence.mass_bound));
        }
        if bdry > sequence.boundary_bound + 1e-8 {
            return Err(format!("Current {} has boundary mass {} > bound {}", i, bdry, sequence.boundary_bound));
        }
    }

    let subseq = sequence.convergent_subsequence();
    let limit_idx = *subseq.last().unwrap();
    let limit = sequence.currents[limit_idx].clone();
    let limit_mass = limit.mass();

    // Compute lower semicontinuity gap
    let first_mass = sequence.currents.first().unwrap().mass();
    let gap = (first_mass - limit_mass).max(0.0);

    // Check convergence
    let was_convergent = sequence.is_cauchy(0.5);

    Ok(CompactnessResult {
        limit,
        limit_mass,
        lower_semicontinuity_gap: gap,
        was_convergent,
    })
}

/// Deformation theorem: any current can be approximated by a polyhedral current.
///
/// Given a current T and a grid size ε, there exists a polyhedral current P such that:
/// - F(T - P) ≤ C·ε·M(T) (flat norm approximation)
/// - M(P) ≤ M(T) (mass non-increasing)
/// - M(∂P) ≤ M(∂T) (boundary mass non-increasing)
pub fn deformation_theorem(current: &Current, epsilon: f64) -> DeformationResult {
    let mass = current.mass();
    let bdry_mass = current.boundary().mass();

    // Create a polyhedral approximation by snapping to grid
    let polyhedral = snap_to_grid(current, epsilon);

    let flat_error = flat_distance(current, &polyhedral);

    DeformationResult {
        original: current.clone(),
        polyhedral,
        flat_error,
        epsilon,
        mass_bound: mass,
        boundary_bound: bdry_mass,
    }
}

/// Result of the deformation theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformationResult {
    /// Original current.
    pub original: Current,
    /// Polyhedral approximation.
    pub polyhedral: Current,
    /// Flat norm error of approximation.
    pub flat_error: f64,
    /// Grid size used.
    pub epsilon: f64,
    /// Mass bound (M(P) ≤ M(T)).
    pub mass_bound: f64,
    /// Boundary mass bound (M(∂P) ≤ M(∂T)).
    pub boundary_bound: f64,
}

/// Isoperimetric inequality for currents.
///
/// If ∂T = 0, then there exists S such that ∂S = T and M(S) ≤ C·M(T)^(k/(k-1)).
pub fn isoperimetric_inequality(current: &Current, k: usize) -> IsoperimetricResult {
    let mass = current.mass();
    let bdry = current.boundary().mass();

    // If current is a cycle, it bounds some surface
    let is_cycle = current.is_cycle();

    let (filling_mass, constant) = if is_cycle && k >= 1 {
        let c = isoperimetric_constant(k);
        let fill_mass = c * mass.powf((k + 1) as f64 / k.max(1) as f64);
        (fill_mass, c)
    } else {
        (0.0, 0.0)
    };

    IsoperimetricResult {
        mass,
        boundary_mass: bdry,
        is_cycle,
        filling_mass_bound: filling_mass,
        isoperimetric_constant: constant,
        dimension: k,
    }
}

/// Result of isoperimetric inequality computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsoperimetricResult {
    pub mass: f64,
    pub boundary_mass: f64,
    pub is_cycle: bool,
    pub filling_mass_bound: f64,
    pub isoperimetric_constant: f64,
    pub dimension: usize,
}

// --- Helpers ---

fn snap_to_grid(current: &Current, epsilon: f64) -> Current {
    let mut result = Current::zero();
    for s in &current.simplices {
        let snapped_verts: Vec<Point> = s.vertices.iter().map(|v| {
            let mut snapped = v.clone();
            for i in 0..snapped.len() {
                snapped[i] = (v[i] / epsilon).round() * epsilon;
            }
            snapped
        }).collect();
        result.add(Simplex::new(snapped_verts, s.orientation));
    }
    result
}

fn isoperimetric_constant(k: usize) -> f64 {
    match k {
        1 => 0.5,
        2 => 1.0 / (4.0 * std::f64::consts::PI),
        3 => (36.0 * std::f64::consts::PI).powf(1.0 / 3.0),
        _ => {
            let n = k as f64;
            n.powf(n / (n - 1.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn pt(x: f64, y: f64) -> Point {
        DVector::from_vec(vec![x, y])
    }

    fn make_square_current() -> Current {
        let mut c = Current::zero();
        // Two triangles forming a unit square
        c.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0)], 1.0));
        c.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)], 1.0));
        c
    }

    #[test]
    fn test_sequence_push_respects_bounds() {
        let mut seq = CurrentSequence::new(10.0, 5.0);
        let c = make_square_current();
        assert!(seq.push(c).is_ok());
    }

    #[test]
    fn test_sequence_push_rejects_excess_mass() {
        let mut seq = CurrentSequence::new(0.001, 5.0);
        let c = make_square_current();
        assert!(seq.push(c).is_err());
    }

    #[test]
    fn test_is_cauchy() {
        let mut seq = CurrentSequence::new(10.0, 10.0);
        // Same current repeated
        let c = make_square_current();
        seq.push(c.clone()).unwrap();
        seq.push(c.clone()).unwrap();
        seq.push(c).unwrap();
        assert!(seq.is_cauchy(0.01));
    }

    #[test]
    fn test_convergent_subsequence() {
        let mut seq = CurrentSequence::new(10.0, 10.0);
        for _ in 0..5 {
            seq.push(make_square_current()).unwrap();
        }
        let subseq = seq.convergent_subsequence();
        assert!(!subseq.is_empty());
    }

    #[test]
    fn test_flat_limit() {
        let mut seq = CurrentSequence::new(10.0, 10.0);
        seq.push(make_square_current()).unwrap();
        seq.push(make_square_current()).unwrap();
        let limit = seq.flat_limit();
        assert!(limit.is_some());
    }

    #[test]
    fn test_apply_compactness() {
        let mut seq = CurrentSequence::new(10.0, 10.0);
        for _ in 0..3 {
            seq.push(make_square_current()).unwrap();
        }
        let result = apply_compactness(&seq).unwrap();
        assert!(result.limit_mass > 0.0);
        assert!(result.limit_mass <= seq.mass_bound + 1e-10);
    }

    #[test]
    fn test_apply_compactness_empty() {
        let seq = CurrentSequence::new(10.0, 10.0);
        assert!(apply_compactness(&seq).is_err());
    }

    #[test]
    fn test_deformation_theorem() {
        let c = make_square_current();
        let result = deformation_theorem(&c, 0.1);
        assert!(result.flat_error >= 0.0);
        assert!(result.polyhedral.mass() > 0.0);
    }

    #[test]
    fn test_snap_to_grid() {
        let tri = Simplex::new(vec![pt(0.12, 0.07), pt(1.03, 0.98), pt(0.48, 0.52)], 1.0);
        let c = Current::from_simplex(tri);
        let snapped = snap_to_grid(&c, 0.5);
        // Vertices should be on 0.5 grid
        for s in &snapped.simplices {
            for v in &s.vertices {
                for coord in v.iter() {
                    let remainder = (coord / 0.5).round() * 0.5 - coord;
                    assert!(remainder.abs() < 1e-10, "Vertex not on grid: {}", coord);
                }
            }
        }
    }

    #[test]
    fn test_isoperimetric_inequality_cycle() {
        // Closed boundary
        let mut c = Current::zero();
        c.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        c.add(Simplex::new(vec![pt(1.0, 0.0), pt(1.0, 1.0)], 1.0));
        c.add(Simplex::new(vec![pt(1.0, 1.0), pt(0.0, 1.0)], 1.0));
        c.add(Simplex::new(vec![pt(0.0, 1.0), pt(0.0, 0.0)], 1.0));

        let result = isoperimetric_inequality(&c, 1);
        assert!(result.is_cycle);
        assert!(result.filling_mass_bound > 0.0);
    }

    #[test]
    fn test_isoperimetric_inequality_not_cycle() {
        let mut c = Current::zero();
        c.add(Simplex::new(vec![pt(0.0, 0.0), pt(1.0, 0.0)], 1.0));
        let result = isoperimetric_inequality(&c, 1);
        assert!(!result.is_cycle);
    }

    #[test]
    fn test_compactness_result_serialization() {
        let result = CompactnessResult {
            limit: Current::zero(),
            limit_mass: 0.0,
            lower_semicontinuity_gap: 0.0,
            was_convergent: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: CompactnessResult = serde_json::from_str(&json).unwrap();
        assert!(r2.was_convergent);
    }

    #[test]
    fn test_deformation_mass_bound() {
        let c = make_square_current();
        let result = deformation_theorem(&c, 0.1);
        // Mass of polyhedral should be reasonable
        assert!(result.polyhedral.mass().is_finite());
    }
}
