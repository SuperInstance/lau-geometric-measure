# lau-geometric-measure

**Geometric measure theory for agents — measuring agent state spaces where classical measure theory isn't fine enough**

A Rust library implementing geometric measure theory: Hausdorff measure, Hausdorff dimension, rectifiability, currents, varifolds, the Plateau problem, Federer-Fleming compactness, isoperimetric inequalities, monotonicity formulas, and agent state-space geometry analysis.

112 tests · 10 modules · ~3,900 LOC

---

## What This Does

Classical measure theory (Lebesgue measure) can't distinguish sets of fractional dimension — a curve has Lebesgue measure zero in the plane, and so does a Cantor set, but they're very different. **Geometric measure theory** provides the finer tools needed:

- **Hausdorff measure** — the `s`-dimensional "size" of a set, for any real `s ≥ 0`
- **Hausdorff dimension** — the critical dimension where measure jumps from ∞ to 0
- **Currents** — oriented `k`-dimensional surfaces as linear functionals on differential forms (Federer-Fleming theory)
- **Varifolds** — unoriented surfaces with tangent structure, for boundaries without consistent orientation
- **Rectifiability** — when a fractal set is actually "almost a manifold" (covered by Lipschitz images of ℝᵏ)
- **Plateau problem** — find the minimal surface spanning a given boundary (soap films)
- **Isoperimetric inequalities** — area bounds from boundary measure
- **Monotonicity formulas** — blow-up analysis and singularity classification for minimal surfaces
- **Agent state space analysis** — measure the effective dimension and structure of agent state spaces

---

## Key Idea

Geometric measure theory provides the mathematical framework for measuring things that classical measure theory can't handle:

1. **Fractal sets** have non-integer dimension (e.g., the Cantor set has dimension log 2 / log 3 ≈ 0.631)
2. **Currents** represent surfaces as objects you can differentiate (∂² = 0) and integrate, even when they're singular
3. **The Plateau problem** has a solution because currents with bounded mass have compact limits (Federer-Fleming compactness)

For agents, this means you can measure the **effective dimension** of a state space from trajectory samples, detect when an agent's behavior regime-shifts, and quantify coverage of exploration.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-geometric-measure = { git = "https://github.com/SuperInstance/lau-geometric-measure" }
```

### Dependencies

- `nalgebra` 0.33 (with `serde-serialize`) — linear algebra
- `serde` 1.x (with `derive`) — serialization
- `num-traits` 0.2 — numeric traits
- `approx` 0.5 — floating-point comparison

---

## Quick Start

```rust
use lau_geometric_measure::prelude::*;
use nalgebra::DVector;

// Hausdorff measure and dimension of a point cloud
let points: Vec<DVector<f64>> = /* agent trajectory samples */;
let dim_result = hausdorff_dimension_auto(&points);
println!("Hausdorff dimension: {:.3} (confidence: {:.2})",
    dim_result.dimension, dim_result.confidence);

// Measure at a specific scale
let measure = hausdorff_measure(&points, dim_result.dimension, 0.1);
println!("H^{} measure: {:.3}", measure.dimension, measure.measure);

// Rectifiability test — is the state space "almost a manifold"?
let rect_result = test_rectifiability(&points, 2);
println!("Rectifiable: {:?}", rect_result.classification);

// Build a current (oriented surface) from the trajectory
let current = state_space_as_current(&points);
println!("Mass: {}, is cycle: {}", current.mass(), current.is_cycle());

// Build a varifold (unoriented surface with tangent structure)
let varifold = state_space_as_varifold(&points, 2, 0.5);

// Full agent state space analysis
let geometry = measure_agent_state_space(&points, 2);
println!("Dimension: {:.2}, rectifiable: {}, mass: {:.3}",
    geometry.hausdorff_dimension,
    geometry.is_rectifiable,
    geometry.total_mass);

// Fractal dimensions of known sets
assert!((fractal_dimension(FractalType::CantorSet) - 0.6309).abs() < 0.01);
assert!((fractal_dimension(FractalType::SierpinskiTriangle) - 1.585).abs() < 0.01);
```

---

## API Reference

### `hausdorff` — Hausdorff Measure & Dimension

| Function | Description |
|---|---|
| `hausdorff_measure(points, s, ε)` | Compute H^s_ε via greedy ball covering |
| `hausdorff_measure_multiscale(points, s, εs)` | Measure at multiple resolutions |
| `hausdorff_dimension(points, ε_min, ε_max, n)` | Estimate dim_H via log-log regression |
| `hausdorff_dimension_auto(points)` | Auto-scale dimension estimation |
| `fractal_dimension(FractalType)` | Known fractal dimensions (Cantor, Sierpinski, Koch, Menger, etc.) |
| `box_counting_dimension(points, ...)` | Box-counting dimension estimate |
| `volume_unit_ball(s)` | Volume of the unit ball in dimension `s` |
| `FractalType` enum | `CantorSet`, `SierpinskiTriangle`, `SierpinskiCarpet`, `KochCurve`, `MengerSponge`, `DragonCurve`, `BrownianPath` |

### `currents` — Oriented Surfaces (Federer-Fleming)

| Type / Function | Description |
|---|---|
| `Simplex` | An oriented k-simplex in n-dimensional space |
| `Current` | A weighted sum of simplices (a k-current) |
| `Current::boundary()` | ∂T — the boundary current |
| `Current::mass()` | Total mass (weighted volume) |
| `Current::flat_norm()` | Flat norm of the current |
| `Current::is_cycle()` | Check if ∂T = 0 |
| `Current::scale(f)` | Scale the current by a factor |
| `Current::translate(v)` | Translate by a vector |
| `Current::pushforward(M)` | Push forward by a linear map |
| `flat_distance(T1, T2)` | Distance between currents in flat norm |
| `verify_boundary_of_boundary_zero(T)` | Verify ∂² = 0 |
| `triangulate_1d(points)` | Build a 1-current from a point sequence |
| `triangulate_2d_fan(points)` | Build a 2-current via fan triangulation |

### `varifolds` — Unoriented Surfaces

| Type / Function | Description |
|---|---|
| `Varifold` | A measure on (point, tangent_plane) pairs |
| `VarifoldElement` | A single (point, tangent basis, weight) entry |
| `Varifold::from_simplices(simplices)` | Build from a simplicial complex |
| `Varifold::mass()` | Total weight |
| `Varifold::first_variation(vf)` | First variation under a vector field |
| `Varifold::mean_curvature()` | Mean curvature vector at each element |
| `Varifold::density_ratio(center, r, k)` | Density ratio θ(x, r) |
| `Varifold::check_monotonicity(center, k, radii)` | Verify monotonicity formula |
| `varifold_from_pointcloud(points, k, r)` | Build varifold from point samples |

### `rectifiability` — Set Structure Classification

| Type / Function | Description |
|---|---|
| `Rectifiability` enum | `Rectifiable`, `Unrectifiable`, `Partial` |
| `test_rectifiability(points, k)` | Classify a point set |
| `check_lipschitz_graph(points, k, ε)` | Test if set is a Lipschitz graph over ℝᵏ |
| `decompose_rectifiability(points, k)` | Split into rectifiable + unrectifiable parts |
| `compute_density(points, center, r)` | k-dimensional density at a point |

### `plateau` — Minimal Surfaces

| Type / Function | Description |
|---|---|
| `PlateauSolution` | Solution to the discrete Plateau problem |
| `solve_plateau(boundary, max_iter)` | Find minimal surface spanning a boundary current |
| `solve_plateau_varifold(boundary, max_iter)` | Minimal surface as varifold |
| `compare_to_isoperimetric_bound(surface, k)` | Compare area to isoperimetric lower bound |
| `minimal_disk_area(radius)` | Area of a minimal disk (π r²) |
| `estimate_minimal_area(boundary)` | Estimate lower bound on minimal area |

### `isoperimetric` — Isoperimetric Inequalities

| Function | Description |
|---|---|
| `isoperimetric_constant(k)` | Classical constant C(k) for ℝᵏ |
| `check_isoperimetric(current, k)` | Verify inequality: Area ≥ C · Boundary^α |
| `sobolev_constant(n, k)` | Sobolev embedding constant |
| `cheeger_constant(points, k)` | Cheeger isoperimetric constant |
| `isoperimetric_profile(points, k)` | Profile function I(v) |

### `monotonicity` — Blow-up Analysis

| Type / Function | Description |
|---|---|
| `MonotonicityResult` | Result of monotonicity formula check |
| `check_monotonicity_current(T, x, radii, k)` | Verify monotonicity for a current |
| `check_monotonicity_varifold(V, x, radii, k)` | Verify for a varifold |
| `blow_up(T, x, r)` | Blow up a current at a point and scale |
| `multi_scale_blowup(T, x, radii)` | Multi-scale blow-up analysis |
| `is_cone(T, x)` | Check if surface is a cone over a point |
| `tangent_cone(T, x)` | Compute the tangent cone at a point |
| `classify_singularity(T, x)` | Classify singularity type |

### `compactness` — Federer-Fleming Compactness

| Type / Function | Description |
|---|---|
| `CurrentSequence` | A sequence of currents with mass/boundary bounds |
| `apply_compactness(sequence)` | Apply compactness theorem: extract convergent subsequence |
| `deformation_theorem(current, ε)` | Deformation theorem: approximate current on ε-grid |
| `CurrentSequence::is_cauchy(tolerance)` | Check if sequence is Cauchy in flat norm |
| `CurrentSequence::flat_limit()` | Compute limit current |

### `agent` — Agent State Space Geometry

| Type / Function | Description |
|---|---|
| `AgentStateSpaceGeometry` | Full geometry description of agent state space |
| `measure_agent_state_space(points, k)` | Compute dimension, rectifiability, mass |
| `detect_regime_changes(points, window, k)` | Detect when agent behavior shifts dimension |
| `exploration_coverage(points, ε, k)` | How thoroughly the state space is explored |
| `compare_state_spaces(points1, points2)` | Compare two state spaces |
| `StateSpaceComparison` | Comparison result with dimension difference and overlap |

---

## How It Works

### Architecture

```
hausdorff (H^s measure, dim_H estimation)
    └── rectifiability (Lipschitz structure analysis)
currents (oriented k-surfaces: simplices, boundary, flat norm)
    ├── varifolds (unoriented surfaces: tangent structure, mean curvature)
    ├── plateau (minimal surface solver)
    ├── isoperimetric (area-bounding inequalities)
    ├── monotonicity (density ratio, blow-up, tangent cones)
    └── compactness (Federer-Fleming, deformation theorem)
agent (state space geometry from trajectory samples)
```

### Hausdorff Measure Approximation

The `s`-dimensional Hausdorff measure is:

$$\mathcal{H}^s_\varepsilon(S) = \inf\left\{\sum_i \left(\frac{\text{diam}(U_i)}{2}\right)^s : S \subseteq \bigcup U_i, \text{diam}(U_i) \le \varepsilon\right\}$$

We approximate this with **greedy ball covering**: iteratively pick the uncovered point farthest from existing balls, place a ball of radius ε, and count balls. The dimension is estimated by **log-log regression** of N(ε) vs 1/ε.

### Currents and the Boundary Operator

A **k-current** is represented as a weighted sum of oriented k-simplices. The boundary operator ∂ maps k-currents to (k-1)-currents, with the fundamental property **∂² = 0** (boundary of boundary is zero). This enables:

- **Mass**: `M(T) = Σ |weight_i| · volume(simplex_i)`
- **Flat norm**: `F(T) = inf { M(A) + M(B) : T = A + ∂B }`
- **Pushforward**: applying a linear map to every simplex

### Varifold First Variation

A varifold's **first variation** δV(X) under a vector field X measures how the surface area changes under the flow of X. Stationary varifolds (minimal surfaces) satisfy δV = 0. The **mean curvature** is extracted from the first variation.

---

## The Math

### Hausdorff Measure

The `s`-dimensional Hausdorff measure generalizes length (s=1), area (s=2), and volume (s=3) to non-integer dimensions. The **Hausdorff dimension** dim_H(S) is the infimum of s where H^s(S) = 0.

### Federer-Fleming Compactness

The space of integral currents with M(T) ≤ A and M(∂T) ≤ B is **compact** in the flat norm topology. This guarantees that minimal surfaces exist as limits — the key to solving the Plateau problem.

### Isoperimetric Inequality

For a k-dimensional surface T with boundary ∂T in ℝⁿ:

$$\text{Area}(T) \ge C(k, n) \cdot \text{Measure}(\partial T)^{k/(k-1)}$$

The classical case (k=2, n=2): Area ≥ Perimeter² / (4π), with equality for disks.

### Monotonicity Formula

For a minimal k-dimensional surface M and any point x:

$$\theta^k(x, r) = \frac{\text{Area}(B_r(x) \cap M)}{\omega_k r^k}$$

is **non-decreasing** in r. Equality between two radii implies M is a cone over x near those scales. This is the fundamental tool for studying singularities of minimal surfaces.

### Rectifiability

A set E is **k-rectifiable** if H^k(E \ ∪ f_i(ℝᵏ)) = 0 for countably many Lipschitz maps f_i. The **structure theorem** says any H^k-measurable set decomposes uniquely into a rectifiable part and a purely unrectifiable part.

---

## Running Tests

```bash
cargo test
```

112 tests across all modules: Hausdorff measure validation against known fractals, boundary operator properties, Plateau problem convergence, isoperimetric bounds, monotonicity verification, and agent state space analysis.

---

## License

MIT
