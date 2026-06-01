//! # lau-geometric-measure
//!
//! Geometric measure theory for agents — measuring agent state spaces where
//! classical measure theory isn't fine enough.
//!
//! Provides tools for:
//! - Hausdorff measure and fractional dimension
//! - Rectifiability of agent manifolds
//! - Currents and varifolds (oriented/unoriented surfaces)
//! - Plateau problem (minimal surfaces)
//! - Federer-Fleming compactness
//! - Isoperimetric inequalities
//! - Agent state space dimension measurement

pub mod hausdorff;
pub mod rectifiability;
pub mod currents;
pub mod varifolds;
pub mod plateau;
pub mod monotonicity;
pub mod compactness;
pub mod isoperimetric;
pub mod agent;

pub mod prelude {
    pub use crate::hausdorff::*;
    pub use crate::rectifiability::*;
    pub use crate::currents::*;
    pub use crate::varifolds::*;
    pub use crate::plateau::*;
    pub use crate::monotonicity::*;
    pub use crate::compactness::*;
    pub use crate::isoperimetric::*;
    pub use crate::agent::*;
}

/// Re-export nalgebra types commonly used.
pub use nalgebra::{DMatrix, DVector, Dyn, Const};
