//! Main simplification pipeline: neatify and its sub-steps.
//!
//! Ports Python `neatnet.simplify`: `neatify`, `neatify_loop`,
//! `neatify_singletons`, `neatify_pairs`, `neatify_clusters`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use geo::{Area, BooleanOps, BoundingRect, Centroid, Contains, Distance, Euclidean, Intersects, Length, Relate, Simplify};
use geo_types::{Coord, Line, LineString, MultiPolygon, Point, Polygon};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressFinish, ProgressStyle};

use crate::artifacts;
use crate::continuity;
use crate::geometry;
use crate::nodes;
use crate::ops;
use crate::types::{EdgeStatus, NeatifyParams, StreetNetwork};

/// Minimal stderr wrapper that implements [`indicatif::TermLike`].
///
/// Renders progress bars to stderr even without a TTY. Overwrites
/// the current line with `\r` on each update.
#[derive(Debug)]
struct StderrTarget;

impl indicatif::TermLike for StderrTarget {
    fn width(&self) -> u16 { 80 }
    fn move_cursor_up(&self, _n: usize) -> std::io::Result<()> { Ok(()) }
    fn move_cursor_down(&self, _n: usize) -> std::io::Result<()> { Ok(()) }
    fn move_cursor_right(&self, _n: usize) -> std::io::Result<()> { Ok(()) }
    fn move_cursor_left(&self, _n: usize) -> std::io::Result<()> { Ok(()) }
    fn write_line(&self, s: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut err = std::io::stderr().lock();
        err.write_all(s.as_bytes())?;
        err.write_all(b"\n")?;
        err.flush()
    }
    fn write_str(&self, s: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut err = std::io::stderr().lock();
        err.write_all(b"\r")?;
        err.write_all(s.as_bytes())?;
        err.flush()
    }
    fn clear_line(&self) -> std::io::Result<()> { Ok(()) }
    fn flush(&self) -> std::io::Result<()> {
        use std::io::Write;
        std::io::stderr().flush()
    }
}

/// Create a styled progress bar that redraws at most 4 times/sec.
fn make_progress_bar(len: u64, label: &str) -> ProgressBar {
    let target = ProgressDrawTarget::term_like_with_hz(Box::new(StderrTarget), 4);
    let pb = ProgressBar::with_draw_target(Some(len), target)
        .with_finish(ProgressFinish::Abandon);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "  {label} [{{bar:30}}] {{pos}}/{{len}} ({{elapsed}})"
        ))
            .unwrap()
            .progress_chars("##-"),
    );
    pb
}

/// Finish a progress bar and emit a newline so it persists on screen.
fn finish_progress_bar(pb: &ProgressBar, label: &str) {
    let len = pb.length().unwrap_or(0);
    let elapsed = pb.elapsed();
    pb.finish_and_clear();
    eprintln!("  {label} {len}/{len} ({elapsed:.1?})");
}

/// Summary returned by [`diagnostics()`] — cheap pre-flight metrics.
#[derive(Debug, Clone)]
pub struct NeatifyDiagnostics {
    pub n_input_edges: usize,
    pub n_edges_after_topology: usize,
    pub n_artifacts: usize,
    pub fai_threshold: f64,
    pub n_singles: usize,
    pub n_pairs: usize,
    pub n_cluster_artifacts: usize,
    pub n_cluster_groups: usize,
    pub max_cluster_size: usize,
    pub fix_topology_secs: f64,
    pub consolidate_secs: f64,
    pub artifact_detection_secs: f64,
}

/// Diagnostic / dry-run mode: runs the cheap setup phases (fix_topology,
/// consolidate_nodes, artifact detection, classification) and returns
/// complexity metrics without performing any simplification.
///
/// Covers ~1-5% of total `neatify` runtime but reveals all the signals
/// needed to estimate cost.
pub fn diagnostics(
    network: &mut StreetNetwork,
    params: &NeatifyParams,
    exclusion_mask: Option<&[Polygon<f64>]>,
) -> Result<NeatifyDiagnostics, NeatifyError> {
    let n_input_edges = network.geometries.len();

    // Step 1: Fix topology
    let t_step = Instant::now();
    let (fixed_geoms, fixed_statuses, fixed_parents) =
        nodes::fix_topology(
            std::mem::take(&mut network.geometries),
            std::mem::take(&mut network.statuses),
            std::mem::take(&mut network.parent_ids),
            params.eps,
        );
    network.geometries = fixed_geoms;
    network.statuses = fixed_statuses;
    network.parent_ids = fixed_parents;
    let fix_topology_secs = t_step.elapsed().as_secs_f64();
    let n_edges_after_topology = network.geometries.len();

    // Step 2: Consolidate nodes
    let t_step = Instant::now();
    let edge_tree = crate::spatial::build_rtree(&network.geometries);
    let (consol_geoms, consol_statuses, consol_parents) = nodes::consolidate_nodes_with_tree(
        &network.geometries,
        &network.statuses,
        &network.parent_ids,
        params.max_segment_length * 2.1,
        false,
        Some(&edge_tree),
    );
    network.geometries = consol_geoms;
    network.statuses = consol_statuses;
    network.parent_ids = consol_parents;
    let consolidate_secs = t_step.elapsed().as_secs_f64();

    // Step 3: Detect artifacts
    let t_step = Instant::now();
    let artifacts = artifacts::get_artifacts(
        &network.geometries,
        params.artifact_threshold,
        params.artifact_threshold_fallback,
        exclusion_mask,
        params.area_threshold_blocks,
        params.isoareal_threshold_blocks,
        params.area_threshold_circles,
        params.isoareal_threshold_circles_enclosed,
        params.isoperimetric_threshold_circles_touching,
    );
    let artifact_detection_secs = t_step.elapsed().as_secs_f64();

    let (artifact_geoms, _artifact_fais, threshold) = match artifacts {
        Some(a) => a,
        None => {
            return Ok(NeatifyDiagnostics {
                n_input_edges,
                n_edges_after_topology,
                n_artifacts: 0,
                fai_threshold: 0.0,
                n_singles: 0,
                n_pairs: 0,
                n_cluster_artifacts: 0,
                n_cluster_groups: 0,
                max_cluster_size: 0,
                fix_topology_secs,
                consolidate_secs,
                artifact_detection_secs,
            });
        }
    };

    // Step 4: Classify artifacts (same logic as neatify_loop)
    let adjacency = artifacts::build_contiguity_graph(&artifact_geoms, true);
    let comp_labels = artifacts::component_labels_from_adjacency(&adjacency);

    let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
    for &label in &comp_labels {
        *comp_sizes.entry(label).or_default() += 1;
    }

    let mut n_singles = 0usize;
    let mut n_pairs = 0usize;
    let mut n_cluster_artifacts = 0usize;
    let mut cluster_labels: HashSet<usize> = HashSet::new();

    for &label in &comp_labels {
        match comp_sizes.get(&label) {
            Some(&1) => n_singles += 1,
            Some(&2) => n_pairs += 1,
            Some(_) => {
                n_cluster_artifacts += 1;
                cluster_labels.insert(label);
            }
            None => {}
        }
    }

    let n_cluster_groups = cluster_labels.len();
    let max_cluster_size = cluster_labels
        .iter()
        .map(|l| comp_sizes[l])
        .max()
        .unwrap_or(0);

    Ok(NeatifyDiagnostics {
        n_input_edges,
        n_edges_after_topology,
        n_artifacts: artifact_geoms.len(),
        fai_threshold: threshold,
        n_singles,
        n_pairs,
        n_cluster_artifacts,
        n_cluster_groups,
        max_cluster_size,
        fix_topology_secs,
        consolidate_secs,
        artifact_detection_secs,
    })
}

/// Top-level simplification entry point.
///
/// Follows the Adaptive Continuity-Preserving Simplification algorithm:
/// 1. CRS validation
/// 2. Topology fixing (induce nodes, remove degree-2 nodes, dedup)
/// 3. Node consolidation (hierarchical clustering)
/// 4. Artifact detection (polygonize → FAI → KDE threshold)
/// 5. Iterative simplification loops:
///    a. Remove dangles within artifacts
///    b. Simplify singletons
///    c. Simplify pairs
///    d. Simplify clusters
///
/// Mirrors Python `neatify()`.
pub fn neatify(
    network: &mut StreetNetwork,
    params: &NeatifyParams,
    exclusion_mask: Option<&[Polygon<f64>]>,
) -> Result<(), NeatifyError> {
    // Step 1: Fix topology
    let t_step = Instant::now();
    let (fixed_geoms, fixed_statuses, fixed_parents) =
        nodes::fix_topology(
            std::mem::take(&mut network.geometries),
            std::mem::take(&mut network.statuses),
            std::mem::take(&mut network.parent_ids),
            params.eps,
        );
    network.geometries = fixed_geoms;
    network.statuses = fixed_statuses;
    network.parent_ids = fixed_parents;
    eprintln!("  fix_topology: {:.1}s ({} edges)", t_step.elapsed().as_secs_f64(), network.geometries.len());

    // Step 2: Consolidate nodes (pass pre-built tree to avoid redundant build)
    let t_step = Instant::now();
    let edge_tree = crate::spatial::build_rtree(&network.geometries);
    let (consol_geoms, consol_statuses, consol_parents) = nodes::consolidate_nodes_with_tree(
        &network.geometries,
        &network.statuses,
        &network.parent_ids,
        params.max_segment_length * 2.1,
        false,
        Some(&edge_tree),
    );
    network.geometries = consol_geoms;
    network.statuses = consol_statuses;
    network.parent_ids = consol_parents;
    eprintln!("  consolidate_nodes: {:.1}s ({} edges)", t_step.elapsed().as_secs_f64(), network.geometries.len());

    // Step 3: Detect artifacts (with iterative expansion)
    let t_step = Instant::now();
    let artifacts = artifacts::get_artifacts(
        &network.geometries,
        params.artifact_threshold,
        params.artifact_threshold_fallback,
        exclusion_mask,
        params.area_threshold_blocks,
        params.isoareal_threshold_blocks,
        params.area_threshold_circles,
        params.isoareal_threshold_circles_enclosed,
        params.isoperimetric_threshold_circles_touching,
    );

    let (artifact_geoms, _artifact_fais, threshold) = match artifacts {
        Some(a) => a,
        None => {
            log::warn!("No artifacts detected. Returning after topology fixes.");
            return Ok(());
        }
    };
    eprintln!("  get_artifacts: {:.1}s ({} artifacts)", t_step.elapsed().as_secs_f64(), artifact_geoms.len());

    if artifact_geoms.is_empty() {
        log::warn!("No artifacts found. Returning after topology fixes.");
        return Ok(());
    }

    // Step 4: Iterative simplification loops
    let mut current_artifacts = artifact_geoms;
    for loop_idx in 0..params.n_loops {
        eprintln!("  loop {} ...", loop_idx + 1);
        let t_loop = Instant::now();
        neatify_loop(network, &current_artifacts, params)?;
        eprintln!("  loop {} simplify: {:.1}s", loop_idx + 1, t_loop.elapsed().as_secs_f64());

        // Free artifact polygons before post-loop cleanup.
        // They'll be re-detected if another loop is needed.
        current_artifacts = vec![];

        // Post-loop cleanup: induce nodes + dedup (matches Python)
        // Use owned variant to avoid briefly holding two copies of all geometries
        let t_step = Instant::now();
        let (induced_geoms, induced_statuses, induced_parents) =
            nodes::induce_nodes_owned(
                std::mem::take(&mut network.geometries),
                std::mem::take(&mut network.statuses),
                std::mem::take(&mut network.parent_ids),
                params.eps,
            );
        network.geometries = induced_geoms;
        network.statuses = induced_statuses;
        network.parent_ids = induced_parents;
        dedup_network(network);
        eprintln!("  loop {} post-cleanup: {:.1}s ({} edges)", loop_idx + 1, t_step.elapsed().as_secs_f64(), network.geometries.len());

        // Re-detect artifacts for subsequent loops (or create empty placeholder for last loop)
        if loop_idx < params.n_loops - 1 {
            let t_step = Instant::now();
            let re_artifacts = artifacts::get_artifacts(
                &network.geometries,
                Some(threshold),
                params.artifact_threshold_fallback,
                exclusion_mask,
                params.area_threshold_blocks,
                params.isoareal_threshold_blocks,
                params.area_threshold_circles,
                params.isoareal_threshold_circles_enclosed,
                params.isoperimetric_threshold_circles_touching,
            );
            eprintln!("  loop {} re-detect artifacts: {:.1}s", loop_idx + 1, t_step.elapsed().as_secs_f64());
            match re_artifacts {
                Some((new_geoms, _new_fais, _new_threshold)) => {
                    current_artifacts = new_geoms;
                }
                None => break,
            }
        }
    }

    // Final cleanup: remove degenerate edges (zero-length or near-zero)
    // that may have been produced by consolidation, topology fixing, or
    // skeleton generation. Python's GEOS operations implicitly filter these.
    // Remove in-place to avoid cloning all surviving geometries.
    let eps = params.eps;
    let mut i = 0;
    while i < network.geometries.len() {
        if network.geometries[i].0.len() < 2 || Euclidean.length(&network.geometries[i]) <= eps {
            network.geometries.swap_remove(i);
            network.statuses.swap_remove(i);
            network.parent_ids.swap_remove(i);
        } else {
            i += 1;
        }
    }

    Ok(())
}

/// One iteration of the simplification loop.
///
/// Mirrors Python `neatify_loop()`.
fn neatify_loop(
    network: &mut StreetNetwork,
    artifact_geoms: &[Polygon<f64>],
    params: &NeatifyParams,
) -> Result<(), NeatifyError> {
    // 1. Remove dangles: drop edges fully inside any artifact, then clean
    let t_step = Instant::now();
    let tree = crate::spatial::build_rtree(&network.geometries);
    let mut dangle_indices = HashSet::new();
    for artifact in artifact_geoms {
        let candidates = nodes::envelope_query_indices_pub(&tree, artifact);
        let art_bbox = artifact.bounding_rect();
        for idx in candidates {
            // Fast bbox disjoint check before expensive relate
            if let (Some(lr), Some(ar)) = (network.geometries[idx].bounding_rect(), &art_bbox) {
                if lr.min().x > ar.max().x
                    || lr.max().x < ar.min().x
                    || lr.min().y > ar.max().y
                    || lr.max().y < ar.min().y
                {
                    continue;
                }
            }
            if line_covered_by_polygon_fast(&network.geometries[idx], artifact) {
                dangle_indices.insert(idx);
            }
        }
    }

    let n_dangles = dangle_indices.len();
    if !dangle_indices.is_empty() {
        // Remove in-place via swap_remove in reverse sorted order
        let mut sorted_dangles: Vec<usize> = dangle_indices.into_iter().collect();
        sorted_dangles.sort_unstable();
        for &i in sorted_dangles.iter().rev() {
            if i < network.geometries.len() {
                network.geometries.swap_remove(i);
                network.statuses.swap_remove(i);
                network.parent_ids.swap_remove(i);
            }
        }
    }

    let (cleaned, clean_statuses, cleaned_parents) =
        nodes::remove_interstitial_nodes(
            std::mem::take(&mut network.geometries),
            std::mem::take(&mut network.statuses),
            std::mem::take(&mut network.parent_ids),
        );
    network.geometries = cleaned;
    network.statuses = clean_statuses;
    network.parent_ids = cleaned_parents;
    log::info!("  [loop] remove_dangles + clean: {:.3}s (dropped {}, {} edges remain)", t_step.elapsed().as_secs_f64(), n_dangles, network.geometries.len());

    // 2. Build contiguity graph on artifacts → classify as singles/pairs/clusters
    let t_step = Instant::now();
    let adjacency = artifacts::build_contiguity_graph(artifact_geoms, true);
    let comp_labels = artifacts::component_labels_from_adjacency(&adjacency);

    // Count component sizes
    let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
    for &label in &comp_labels {
        *comp_sizes.entry(label).or_default() += 1;
    }

    // Classify: isolates (size 1), pairs (size 2), clusters (size 3+)
    let mut singles = Vec::new();
    let mut pairs = Vec::new();
    let mut clusters = Vec::new();

    for (i, &label) in comp_labels.iter().enumerate() {
        match comp_sizes.get(&label) {
            Some(&1) => singles.push(i),
            Some(&2) => pairs.push(i),
            Some(_) => clusters.push(i),
            None => {}
        }
    }
    log::info!("  [loop] classify: {:.3}s ({} singles, {} pairs, {} clusters)", t_step.elapsed().as_secs_f64(), singles.len(), pairs.len(), clusters.len());

    // Single progress bar for the entire loop — counts every artifact
    // processed across singletons, pairs, and clusters.
    let total = artifact_geoms.len() as u64;
    let pb = make_progress_bar(total, "simplify");

    // 3. Simplify singletons
    if !singles.is_empty() {
        let t_step = Instant::now();
        neatify_singletons(network, artifact_geoms, &singles, params, None, Some(&pb))?;
        log::info!("  [loop] neatify_singletons: {:.3}s", t_step.elapsed().as_secs_f64());
    }

    // 4. Simplify pairs
    if !pairs.is_empty() {
        let t_step = Instant::now();
        neatify_pairs(network, artifact_geoms, &pairs, &comp_labels, params, Some(&pb))?;
        log::info!("  [loop] neatify_pairs: {:.3}s", t_step.elapsed().as_secs_f64());
    }

    // 5. Simplify clusters
    if !clusters.is_empty() {
        let t_step = Instant::now();
        neatify_clusters(network, artifact_geoms, &clusters, &comp_labels, params, Some(&pb))?;
        log::info!("  [loop] neatify_clusters: {:.3}s", t_step.elapsed().as_secs_f64());
    }

    finish_progress_bar(&pb, "simplify");
    Ok(())
}

/// Simplify singleton face artifacts.
///
/// For each single artifact:
/// 1. Run COINS and CES classification
/// 2. Link nodes to artifacts
/// 3. Dispatch to appropriate handler (n1_g1_identical, nx_gx_identical, nx_gx)
///
/// When `precomputed_coins` is `Some`, the provided COINS result is reused
/// instead of recomputing from scratch. This mirrors the Python
/// `compute_coins=False` optimisation used in the pairs stage.
fn neatify_singletons(
    network: &mut StreetNetwork,
    artifact_geoms: &[Polygon<f64>],
    artifact_indices: &[usize],
    params: &NeatifyParams,
    precomputed_coins: Option<&continuity::CoinsResult>,
    progress: Option<&ProgressBar>,
) -> Result<(), NeatifyError> {
    // Reuse caller-provided COINS result, or compute fresh
    let owned_coins;
    let coins_result = match precomputed_coins {
        Some(c) => c,
        None => {
            owned_coins = continuity::coins(&network.geometries, params.angle_threshold);
            &owned_coins
        }
    };

    // Get CES info for singletons
    let singleton_geoms: Vec<Polygon<f64>> = artifact_indices
        .iter()
        .map(|&i| artifact_geoms[i].clone())
        .collect();
    let ces_info =
        continuity::get_stroke_info(&singleton_geoms, &network.geometries, coins_result);

    // Build R-tree once for all singleton lookups
    let tree = crate::spatial::build_rtree(&network.geometries);

    use rayon::prelude::*;
    let results: Vec<(Vec<usize>, Vec<LineString<f64>>)> = artifact_indices
        .par_iter()
        .enumerate()
        .map(|(local_idx, &art_idx)| {
            let mut local_drop: Vec<usize> = Vec::new();
            let mut local_add: Vec<LineString<f64>> = Vec::new();

            let artifact = &artifact_geoms[art_idx];
            let ces = &ces_info[local_idx];

            let covered_edges = find_covered_edges_with_tree(
                &network.geometries, &tree, artifact, params.eps,
            );
            if covered_edges.is_empty() {
                if let Some(pb) = progress { pb.inc(1); }
                return (local_drop, local_add);
            }

            let covered_geoms: Vec<LineString<f64>> = covered_edges
                .iter()
                .map(|&i| network.geometries[i].clone())
                .collect();
            let (node_coords, _) = nodes::nodes_from_edges(&covered_geoms);
            let n_nodes = node_coords.len();

            let n_strokes = ces.stroke_count;

            if n_strokes > n_nodes {
                if let Some(pb) = progress { pb.inc(1); }
                return (local_drop, local_add);
            }

            if n_nodes == 1 && n_strokes == 1 {
                process_n1_g1_identical(
                    &covered_edges,
                    artifact,
                    &node_coords,
                    &network.geometries,
                    params,
                    &mut local_drop,
                    &mut local_add,
                );
            } else if n_nodes > 1 && is_identical_ces(ces) {
                process_nx_gx_identical(
                    &covered_edges,
                    artifact,
                    &node_coords,
                    &network.geometries,
                    params,
                    &mut local_drop,
                    &mut local_add,
                );
            } else if n_nodes > 1 {
                process_nx_gx(
                    &covered_edges,
                    artifact,
                    &node_coords,
                    &network.geometries,
                    coins_result,
                    params,
                    &mut local_drop,
                    &mut local_add,
                );
            }

            if let Some(pb) = progress { pb.inc(1); }
            (local_drop, local_add)
        })
        .collect();

    let mut to_drop: Vec<usize> = Vec::new();
    let mut to_add: Vec<LineString<f64>> = Vec::new();
    for (drops, adds) in results {
        to_drop.extend(drops);
        to_add.extend(adds);
    }

    apply_changes(network, &to_drop, &to_add, params);
    Ok(())
}

/// Classify the shared edge from one artifact's perspective using COINS groups.
///
/// Mirrors Python `get_type()` (simplify.py:624-657).
/// Returns 'C' (continuing), 'E' (end), or 'S' (standalone/roundabout).
fn get_ces_type(
    covered_indices: &[usize],
    shared_idx: usize,
    coins: &continuity::CoinsResult,
) -> char {
    if covered_indices.is_empty() {
        return 'S';
    }

    // Roundabout special case: all edges in one group, and all group members present
    let groups: HashSet<usize> = covered_indices.iter().map(|&i| coins.group[i]).collect();
    if groups.len() == 1 {
        let first_count = coins.stroke_count[covered_indices[0]];
        if covered_indices.len() == first_count {
            return 'S';
        }
    }

    // Find end groups (groups containing any end edge)
    let end_groups: HashSet<usize> = covered_indices
        .iter()
        .filter(|&&i| coins.is_end[i])
        .map(|&i| coins.group[i])
        .collect();

    // Main edges: those NOT in end groups
    let shared_group = coins.group[shared_idx];
    let shared_is_main = !end_groups.contains(&shared_group);

    if shared_is_main {
        return 'C';
    }

    // Check if all edges of shared's group are within covered_indices
    let shared_count_in_covered = covered_indices
        .iter()
        .filter(|&&i| coins.group[i] == shared_group)
        .count();
    if coins.stroke_count[shared_idx] == shared_count_in_covered {
        return 'S';
    }

    'E'
}

/// Simplify pairs of face artifacts.
///
/// Mirrors Python `neatify_pairs()` with full `get_solution()` dispatch.
fn neatify_pairs(
    network: &mut StreetNetwork,
    artifact_geoms: &[Polygon<f64>],
    artifact_indices: &[usize],
    comp_labels: &[usize],
    params: &NeatifyParams,
    progress: Option<&ProgressBar>,
) -> Result<(), NeatifyError> {
    // Group artifacts by component label into pairs
    let mut pair_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &i in artifact_indices {
        pair_groups.entry(comp_labels[i]).or_default().push(i);
    }

    let coins_result = continuity::coins(&network.geometries, params.angle_threshold);

    // Build R-tree once for all pair lookups
    let tree = crate::spatial::build_rtree(&network.geometries);

    // Non-planar detection: stroke_count > node_count
    // (mirrors Python _identify_non_planar)
    let singleton_geoms: Vec<Polygon<f64>> = artifact_indices
        .iter()
        .map(|&i| artifact_geoms[i].clone())
        .collect();
    let ces_info =
        continuity::get_stroke_info(&singleton_geoms, &network.geometries, &coins_result);
    let mut non_planar: HashSet<usize> = HashSet::new();
    for (local_idx, &art_idx) in artifact_indices.iter().enumerate() {
        let artifact = &artifact_geoms[art_idx];
        let covered = find_covered_edges_with_tree(
            &network.geometries, &tree, artifact, params.eps,
        );
        let covered_geoms: Vec<LineString<f64>> = covered.iter().map(|&i| network.geometries[i].clone()).collect();
        let (node_coords, _) = nodes::nodes_from_edges(&covered_geoms);
        if ces_info[local_idx].stroke_count > node_coords.len() {
            non_planar.insert(art_idx);
        }
    }

    // Determine solution for each pair
    let mut drop_interline_pairs: Vec<(Vec<usize>, usize)> = Vec::new(); // (pair, shared_idx)
    let mut iterate_pairs: Vec<Vec<usize>> = Vec::new();
    let mut skeleton_pairs: Vec<Vec<usize>> = Vec::new();
    let mut planar_from_np_clusters: Vec<usize> = Vec::new();

    for (_label, pair) in &pair_groups {
        if pair.len() != 2 {
            if let Some(pb) = progress { pb.inc(pair.len() as u64); }
            continue;
        }

        // Handle non-planar pairs (mirrors Python component-level bookkeeping)
        let np_count = pair.iter().filter(|&&i| non_planar.contains(&i)).count();
        if np_count > 0 {
            if np_count == 2 {
                // Both non-planar: send to skeleton
                skeleton_pairs.push(pair.clone());
            } else {
                // Mixed pair: collect planar member for singleton processing
                for &idx in pair {
                    if !non_planar.contains(&idx) {
                        planar_from_np_clusters.push(idx);
                    }
                }
            }
            continue;
        }

        // Find edges covered by each artifact (distance-based, no buffer needed)
        let covered_a = find_covered_edges_with_tree(
            &network.geometries, &tree, &artifact_geoms[pair[0]], params.eps,
        );
        let covered_b = find_covered_edges_with_tree(
            &network.geometries, &tree, &artifact_geoms[pair[1]], params.eps,
        );
        let set_a: HashSet<usize> = covered_a.iter().copied().collect();
        let set_b: HashSet<usize> = covered_b.iter().copied().collect();
        let shared: Vec<usize> = set_a.intersection(&set_b).copied().collect();

        // Non-planar: empty shared or empty covers
        if shared.is_empty() || covered_a.is_empty() || covered_b.is_empty() {
            if let Some(pb) = progress { pb.inc(2); }
            continue;
        }

        if shared.len() > 1 {
            // Multiple shared edges → skeleton
            skeleton_pairs.push(pair.clone());
            continue;
        }

        let shared_idx = shared[0];

        // Coverage asymmetry check (mirrors Python):
        // Count edges touching B that are NOT covered by A (and vice versa)
        let b_not_in_a = covered_b.iter().filter(|i| !set_a.contains(i)).count();
        let a_not_in_b = covered_a.iter().filter(|i| !set_b.contains(i)).count();
        if b_not_in_a == 1 || a_not_in_b == 1 {
            drop_interline_pairs.push((pair.clone(), shared_idx));
            continue;
        }

        // CES classification of shared edge from each artifact's perspective
        let seen_by_a = get_ces_type(&covered_a, shared_idx, &coins_result);
        let seen_by_b = get_ces_type(&covered_b, shared_idx, &coins_result);

        if seen_by_a == 'C' && seen_by_b == 'C' {
            iterate_pairs.push(pair.clone());
        } else if seen_by_a == seen_by_b {
            drop_interline_pairs.push((pair.clone(), shared_idx));
        } else {
            skeleton_pairs.push(pair.clone());
        }
    }

    // Drop shared edges for drop_interline pairs before singleton dispatch
    if !drop_interline_pairs.is_empty() {
        let drop_indices: HashSet<usize> = drop_interline_pairs
            .iter()
            .map(|(_, shared_idx)| *shared_idx)
            .collect();
        // Remove in-place via swap_remove in reverse sorted order
        let mut sorted_drops: Vec<usize> = drop_indices.into_iter().collect();
        sorted_drops.sort_unstable();
        for &i in sorted_drops.iter().rev() {
            if i < network.geometries.len() {
                network.geometries.swap_remove(i);
                network.statuses.swap_remove(i);
                network.parent_ids.swap_remove(i);
            }
        }

        // Clean topology after drops
        let (cleaned, clean_statuses, cleaned_parents) =
            nodes::remove_interstitial_nodes(
                std::mem::take(&mut network.geometries),
                std::mem::take(&mut network.statuses),
                std::mem::take(&mut network.parent_ids),
            );
        network.geometries = cleaned;
        network.statuses = clean_statuses;
        network.parent_ids = cleaned_parents;
    }

    // Dissolve drop_interline pairs into single polygons (mirrors Python
    // `artifacts_w_info.query(sol_drop).dissolve("comp", as_index=False)`)
    // and build an extended artifact_geoms that includes the merged polygons.
    let mut extended_geoms: Vec<Polygon<f64>> = artifact_geoms.to_vec();
    let mut dissolved_indices: Vec<usize> = Vec::new();
    for (pair, _) in &drop_interline_pairs {
        let a = &artifact_geoms[pair[0]];
        let b = &artifact_geoms[pair[1]];
        let merged = MultiPolygon(vec![a.clone()]).union(&MultiPolygon(vec![b.clone()]));
        // Use the largest polygon from the union result
        if let Some(largest) = merged.0.into_iter().max_by(|x, y| {
            x.unsigned_area()
                .partial_cmp(&y.unsigned_area())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let idx = extended_geoms.len();
            extended_geoms.push(largest);
            dissolved_indices.push(idx);
        }
    }

    // Process drop_interline (dissolved) + iterate (first pass) + planar from
    // mixed non-planar pairs as singletons
    if !dissolved_indices.is_empty() || !iterate_pairs.is_empty() || !planar_from_np_clusters.is_empty() {
        let mut first_indices: Vec<usize> = dissolved_indices;
        // First pass of iterate pairs
        for pair in &iterate_pairs {
            first_indices.push(pair[0]);
        }
        // Planar artifacts from mixed non-planar pairs
        first_indices.extend(&planar_from_np_clusters);
        if !first_indices.is_empty() {
            // Reuse COINS already computed at pairs level (mirrors Python compute_coins=False)
            neatify_singletons(network, &extended_geoms, &first_indices, params, Some(&coins_result), progress)?;
        }

        // Second pass for iterate pairs – network was modified, recompute COINS
        let second_indices: Vec<usize> = iterate_pairs.iter().map(|p| p[1]).collect();
        if !second_indices.is_empty() {
            neatify_singletons(network, &extended_geoms, &second_indices, params, None, progress)?;
        }
    }

    // Process skeleton pairs via cluster approach
    if !skeleton_pairs.is_empty() {
        let skeleton_indices: Vec<usize> = skeleton_pairs.iter().flatten().copied().collect();
        neatify_clusters(network, artifact_geoms, &skeleton_indices, comp_labels, params, progress)?;
    }

    Ok(())
}

/// Union a slice of polygons using a cascaded tree strategy.
/// Pairwise-unions at each level keep operand complexity balanced: O(n log n)
/// total vertices instead of O(n^2) for sequential accumulation.
fn cascaded_union(polygons: &[Polygon<f64>]) -> MultiPolygon<f64> {
    if polygons.is_empty() {
        return MultiPolygon(vec![]);
    }
    if polygons.len() == 1 {
        return MultiPolygon(vec![polygons[0].clone()]);
    }
    let mut level: Vec<MultiPolygon<f64>> = polygons
        .iter()
        .map(|p| MultiPolygon(vec![p.clone()]))
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut i = 0;
        while i + 1 < level.len() {
            next.push(level[i].union(&level[i + 1]));
            i += 2;
        }
        if i < level.len() {
            next.push(std::mem::replace(&mut level[i], MultiPolygon(vec![])));
        }
        level = next;
    }
    level.into_iter().next().unwrap_or(MultiPolygon(vec![]))
}

/// Split a large cluster of artifact indices into spatially partitioned sub-clusters.
///
/// Uses a grid partition based on polygon centroids. Each grid cell's artifacts
/// become a sub-cluster that can be processed independently. This breaks up
/// monster clusters (e.g., 9,646 artifacts) into manageable parallel work items.
fn split_cluster_spatially(
    cluster: &[usize],
    artifact_geoms: &[Polygon<f64>],
    target_size: usize,
) -> Vec<Vec<usize>> {
    use geo::Centroid;

    // Compute centroids and bounding box
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let centroids: Vec<Option<Point<f64>>> = cluster
        .iter()
        .map(|&idx| {
            let c = artifact_geoms[idx].centroid();
            if let Some(pt) = &c {
                min_x = min_x.min(pt.x());
                min_y = min_y.min(pt.y());
                max_x = max_x.max(pt.x());
                max_y = max_y.max(pt.y());
            }
            c
        })
        .collect();

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    if range_x <= 0.0 || range_y <= 0.0 {
        return vec![cluster.to_vec()];
    }

    // Grid dimensions: aim for target_size artifacts per cell
    let n_cells = (cluster.len() as f64 / target_size as f64).sqrt().ceil() as usize;
    let n_cells = n_cells.max(2);
    let cell_w = range_x / n_cells as f64;
    let cell_h = range_y / n_cells as f64;

    // Assign each artifact to a grid cell
    let mut cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, &art_idx) in cluster.iter().enumerate() {
        let (cx, cy) = match &centroids[i] {
            Some(pt) => (pt.x(), pt.y()),
            None => continue,
        };
        let col = ((cx - min_x) / cell_w).floor() as usize;
        let row = ((cy - min_y) / cell_h).floor() as usize;
        let col = col.min(n_cells - 1);
        let row = row.min(n_cells - 1);
        cells.entry((row, col)).or_default().push(art_idx);
    }

    // Separate large (>= 3) and small (< 3) sub-clusters
    let mut large: Vec<Vec<usize>> = Vec::new();
    let mut orphans: Vec<usize> = Vec::new();
    for cell in cells.into_values() {
        if cell.len() >= 3 {
            large.push(cell);
        } else {
            orphans.extend(cell);
        }
    }

    // If no large sub-clusters exist, return the original cluster unsplit
    if large.is_empty() {
        return vec![cluster.to_vec()];
    }

    // Reassign each orphan to the nearest large sub-cluster by centroid distance
    for orphan_idx in orphans {
        let orphan_centroid = match artifact_geoms[orphan_idx].centroid() {
            Some(pt) => pt,
            None => {
                // No centroid — just put it in the first large sub-cluster
                large[0].push(orphan_idx);
                continue;
            }
        };

        let mut best = 0usize;
        let mut best_dist = f64::INFINITY;
        for (i, sub) in large.iter().enumerate() {
            // Use centroid of the first artifact in the sub-cluster as a representative
            // (computing a full sub-cluster centroid is unnecessary overhead)
            if let Some(pt) = artifact_geoms[sub[0]].centroid() {
                let dx = orphan_centroid.x() - pt.x();
                let dy = orphan_centroid.y() - pt.y();
                let d = dx * dx + dy * dy;
                if d < best_dist {
                    best_dist = d;
                    best = i;
                }
            }
        }
        large[best].push(orphan_idx);
    }

    large
}

/// Simplify clusters of face artifacts.
fn neatify_clusters(
    network: &mut StreetNetwork,
    artifact_geoms: &[Polygon<f64>],
    artifact_indices: &[usize],
    comp_labels: &[usize],
    params: &NeatifyParams,
    progress: Option<&ProgressBar>,
) -> Result<(), NeatifyError> {
    // Group artifacts by component label
    let mut cluster_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in artifact_indices {
        cluster_groups.entry(comp_labels[i]).or_default().push(i);
    }

    let mut to_drop: Vec<usize> = Vec::new();
    let mut to_add: Vec<LineString<f64>> = Vec::new();

    let mut sorted_cluster_labels: Vec<_> = cluster_groups.keys().copied().collect();
    sorted_cluster_labels.sort();

    // Build R-tree once for all cluster lookups
    let tree = crate::spatial::build_rtree(&network.geometries);

    // Split large clusters spatially before processing. This breaks monster
    // clusters (thousands of artifacts) into manageable sub-clusters that
    // produce smaller merged polygons, avoiding the O(n²+) voronoi_skeleton
    // bottleneck on single huge polygons.
    const CLUSTER_SPLIT_THRESHOLD: usize = 500;
    const CLUSTER_SPLIT_TARGET_SIZE: usize = 100;

    let mut work_clusters: Vec<Vec<usize>> = Vec::new();

    for label in &sorted_cluster_labels {
        let cluster = &cluster_groups[label];
        if cluster.len() < 3 { continue; }

        if cluster.len() > CLUSTER_SPLIT_THRESHOLD {
            let sub_clusters = split_cluster_spatially(cluster, artifact_geoms, CLUSTER_SPLIT_TARGET_SIZE);
            log::info!("    [clusters] split cluster of {} artifacts into {} sub-clusters",
                       cluster.len(), sub_clusters.len());
            work_clusters.extend(sub_clusters);
        } else {
            work_clusters.push(cluster.clone());
        }
    }

    let t_clusters = Instant::now();
    log::info!("    [clusters] {} work clusters (after splitting), rayon threads={}",
               work_clusters.len(), rayon::current_num_threads());

    use rayon::prelude::*;

    // Phase 1: Cascaded union for all (sub-)clusters in parallel.
    // Each cluster produces a simplified MultiPolygon and artifact count for progress.
    let merged_clusters: Vec<(MultiPolygon<f64>, u64)> = work_clusters
        .par_iter()
        .map(|cluster| {
            let n_artifacts = cluster.len() as u64;
            let cluster_polys: Vec<Polygon<f64>> = cluster.iter().map(|&i| artifact_geoms[i].clone()).collect();
            let merged = cascaded_union(&cluster_polys);
            if merged.0.is_empty() {
                if let Some(pb) = progress { pb.inc(n_artifacts); }
                return (MultiPolygon::new(vec![]), 0);
            }
            let simplified = MultiPolygon::new(
                merged.0.into_iter()
                    .map(|p| p.simplify(params.eps / 10.0))
                    .collect()
            );
            (simplified, n_artifacts)
        })
        .collect();

    // Phase 2: Flatten all merged polygons into a single work list, then
    // process them in one flat par_iter. This avoids nested parallelism and
    // distributes work evenly — critical when one cluster produces hundreds
    // of merged polygons (e.g., 559 from a 9646-artifact cluster).
    let work_items: Vec<&Polygon<f64>> = merged_clusters
        .iter()
        .flat_map(|(mp, _)| mp.0.iter())
        .collect();

    log::info!("    [clusters] {} merged polygons to process", work_items.len());

    let poly_results: Vec<(Vec<usize>, Vec<LineString<f64>>)> = work_items
        .par_iter()
        .filter_map(|merged_poly| {
            let covered = find_covered_edges_with_tree(&network.geometries, &tree, merged_poly, params.eps);
            if covered.is_empty() {
                return None;
            }

            let boundary_edges =
                find_boundary_edges_with_tree(&network.geometries, &tree, merged_poly, params.eps);
            if boundary_edges.is_empty() {
                return None; // No connections — leave edges untouched (matches Python)
            }

            // Use the polygon exterior ring as skeleton input, split at connection points
            // (where boundary-crossing edges touch the polygon). This produces separate
            // input lines for each "side" of the polygon, so voronoi_skeleton generates
            // proper centerline ridges between opposite sides.
            let tol = params.eps.max(1e-3);
            let connection_pts = find_connection_points(
                &network.geometries, &boundary_edges, merged_poly.exterior(), tol,
            );
            let ring_sides = split_ring_at_points(merged_poly.exterior(), &connection_pts, tol);

            if ring_sides.len() < 2 {
                // Need at least 2 sides for skeleton to produce ridges
                return None;
            }

            // Use small clip_limit since input lines are ON the boundary
            // (matches Python's clip_limit=1e-4 for clusters).
            let (skel, _) = geometry::voronoi_skeleton(
                &ring_sides,
                Some(merged_poly),
                None,
                params.max_segment_length,
                None,
                None,
                1e-4,
                Some(params.consolidation_tolerance),
            );

            // Safeguard 1: if the skeleton is pathologically short, it has collapsed.
            let covered_len: f64 = covered.iter().map(|&i| Euclidean.length(&network.geometries[i])).sum();
            let skel_len: f64 = skel.iter().map(|s| Euclidean.length(s)).sum();
            if covered_len > 0.0 && skel_len < covered_len * 0.1 {
                log::info!("    [clusters] skeleton collapsed ({:.0}m vs {:.0}m covered) — preserving edges",
                    skel_len, covered_len);
                return None;
            }

            // Safeguard 2: verify every connection point has a nearby skeleton endpoint.
            // If any connection point is orphaned, the skeleton would create a gap in
            // the network. Better to not simplify than to disconnect the graph.
            if !connection_pts.is_empty() {
                let connect_tol = params.consolidation_tolerance;
                let skel_endpoints: Vec<Coord<f64>> = skel.iter()
                    .flat_map(|s| {
                        let first = *s.0.first().unwrap();
                        let last = *s.0.last().unwrap();
                        std::iter::once(first).chain(std::iter::once(last))
                    })
                    .collect();

                let orphaned = connection_pts.iter().any(|cp| {
                    !skel_endpoints.iter().any(|sep| {
                        let dx = cp.x - sep.x;
                        let dy = cp.y - sep.y;
                        dx * dx + dy * dy <= connect_tol * connect_tol
                    })
                });

                if orphaned {
                    log::info!(
                        "    [clusters] skeleton disconnects a connection point — preserving edges"
                    );
                    return None;
                }
            }

            Some((covered, skel))
        })
        .collect();

    for (drops, adds) in poly_results {
        to_drop.extend(drops);
        to_add.extend(adds);
    }

    // Update progress for clusters that produced merged polygons
    for (_, n_artifacts) in &merged_clusters {
        if *n_artifacts > 0 {
            if let Some(pb) = progress { pb.inc(*n_artifacts); }
        }
    }

    log::info!("    [clusters] {} work clusters processed in {:.3}s", work_clusters.len(), t_clusters.elapsed().as_secs_f64());

    apply_changes(network, &to_drop, &to_add, params);
    Ok(())
}

// ─── Processing functions ─────────────────────────────────────────────────

/// Process n1_g1_identical: 1 node, 1 stroke group.
///
/// Drop the covered edge and generate voronoi_skeleton replacement.
fn process_n1_g1_identical(
    covered_edges: &[usize],
    artifact: &Polygon<f64>,
    node_coords: &[[f64; 2]],
    geometries: &[LineString<f64>],
    params: &NeatifyParams,
    to_drop: &mut Vec<usize>,
    to_add: &mut Vec<LineString<f64>>,
) {
    let covered_geoms: Vec<LineString<f64>> = covered_edges
        .iter()
        .map(|&i| geometries[i].clone())
        .collect();

    // Build snap targets from node coordinates
    let snap_targets = node_coords_to_lines(node_coords);

    let (edgelines, _splitters) = geometry::voronoi_skeleton(
        &covered_geoms,
        Some(artifact),
        Some(&snap_targets),
        params.max_segment_length,
        None,
        None,
        params.clip_limit,
        Some(params.consolidation_tolerance),
    );

    to_drop.extend(covered_edges);
    let cleaned = remove_dangles(&edgelines, artifact, params.eps);
    to_add.extend(cleaned);
}

/// Process nx_gx_identical: N>1 nodes, all same CES type.
///
/// Drop all covered edges and connect entry points to centroid.
/// If connections aren't within the polygon, use voronoi_skeleton instead.
fn process_nx_gx_identical(
    covered_edges: &[usize],
    artifact: &Polygon<f64>,
    node_coords: &[[f64; 2]],
    geometries: &[LineString<f64>],
    params: &NeatifyParams,
    to_drop: &mut Vec<usize>,
    to_add: &mut Vec<LineString<f64>>,
) {
    // Find relevant nodes (nodes touching the artifact)
    let relevant_nodes: Vec<[f64; 2]> = find_nodes_near_polygon(node_coords, artifact, params.eps);

    if relevant_nodes.is_empty() {
        return;
    }

    // Compute centroid of the artifact
    let centroid = match artifact.centroid() {
        Some(c) => c,
        None => return,
    };
    let cx = centroid.x();
    let cy = centroid.y();

    // Create shortest lines from each relevant node to centroid
    let mut lines = Vec::new();
    let mut all_within = true;

    for node in &relevant_nodes {
        let line = LineString::new(vec![
            Coord { x: node[0], y: node[1] },
            Coord { x: cx, y: cy },
        ]);
        if !geometry::is_within(&line, artifact, 0.1) {
            all_within = false;
            break;
        }
        lines.push(line);
    }

    to_drop.extend(covered_edges);

    if all_within && !lines.is_empty() {
        // Check angle between two lines for sharp angle
        // Use fixed 75° cutoff (not the COINS angle_threshold) to match Python behavior
        if lines.len() == 2 {
            let angle = geometry::angle_between_two_lines(&lines[0], &lines[1]);
            if angle < 75.0 {
                // Replace with direct connection between nodes
                let direct = LineString::new(vec![
                    Coord { x: relevant_nodes[0][0], y: relevant_nodes[0][1] },
                    Coord { x: relevant_nodes[1][0], y: relevant_nodes[1][1] },
                ]);
                to_add.push(direct);
                return;
            }
        }
        to_add.extend(lines);
    } else {
        // Use voronoi_skeleton instead
        let covered_geoms: Vec<LineString<f64>> = covered_edges
            .iter()
            .map(|&i| geometries[i].clone())
            .collect();
        let snap_targets = node_coords_to_lines(&relevant_nodes);

        let (edgelines, _) = geometry::voronoi_skeleton(
            &covered_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            None,
            params.clip_limit,
            Some(params.consolidation_tolerance),
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
    }
}

/// Process nx_gx: N>1 nodes, mixed CES types.
///
/// The most complex case. Classifies covered edges by CES hierarchy (C > E > S),
/// drops lower-hierarchy edges, then reconnects disconnected nodes.
///
/// Key branches:
/// 1. Check if dropping E/S edges causes disconnection (via connected components)
/// 2. If disconnected and multiple C edges: use skeleton snapped to high-degree nodes
/// 3. If disconnected and single C: connect remaining nodes via shortest lines or skeleton
/// 4. Loop special case: single C + single E/S → shortest line if within polygon
/// 5. Sausage special case: 2 nodes + 2 strokes → shortest line between endpoints
fn process_nx_gx(
    covered_edges: &[usize],
    artifact: &Polygon<f64>,
    node_coords: &[[f64; 2]],
    geometries: &[LineString<f64>],
    coins_result: &continuity::CoinsResult,
    params: &NeatifyParams,
    to_drop: &mut Vec<usize>,
    to_add: &mut Vec<LineString<f64>>,
) {
    // Classify edges by CES hierarchy
    let (c_edges, e_edges, s_edges, es_mask, highest) =
        classify_ces_edges(covered_edges, coins_result);

    let n_c = c_edges.len();
    let n_e = e_edges.len();
    let n_s = s_edges.len();
    let n_nodes = node_coords.len();
    let n_strokes = {
        let groups: HashSet<usize> = covered_edges.iter().map(|&i| coins_result.group[i]).collect();
        groups.len()
    };


    // Drop ES edges
    to_drop.extend(&es_mask);

    // Check connected components after dropping ES
    let remaining_geoms: Vec<LineString<f64>> = highest
        .iter()
        .map(|&i| geometries[i].clone())
        .collect();
    let n_comps = if remaining_geoms.is_empty() {
        0
    } else {
        nodes::get_components(&remaining_geoms).iter().collect::<HashSet<_>>().len()
    };

    let relevant_nodes = find_nodes_near_polygon(node_coords, artifact, params.eps);

    // === BRANCH: Loop special case ===
    // C == 1, (E + S) == 1, and a shortest line within the polygon is much shorter than C
    if n_c == 1 && (n_e + n_s) == 1 && relevant_nodes.len() == 2 {
        let shortest = LineString::new(vec![
            Coord { x: relevant_nodes[0][0], y: relevant_nodes[0][1] },
            Coord { x: relevant_nodes[1][0], y: relevant_nodes[1][1] },
        ]);
        let c_len: f64 = c_edges.iter()
            .map(|&i| Euclidean.length(&geometries[i]))
            .sum();
        let s_len = Euclidean.length(&shortest);
        if geometry::is_within(&shortest, artifact, params.eps) && s_len < c_len * 0.5 {
            to_add.push(shortest);
            return;
        }
        // Otherwise fall through to general handling
    }

    // === BRANCH: Sausage special case ===
    // 2 nodes, 2 strokes — just drop E/S, keep C (no reconnection needed)
    if n_nodes == 2 && n_strokes == 2 && !highest.is_empty() {
        return;
    }

    // === BRANCH: All dropped (no C edges) → replace with skeleton ===
    if highest.is_empty() && !es_mask.is_empty() {
        let es_geoms: Vec<LineString<f64>> = es_mask
            .iter()
            .map(|&i| geometries[i].clone())
            .collect();
        let snap_targets = node_coords_to_lines(&relevant_nodes);
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            None,
            params.clip_limit,
            Some(params.consolidation_tolerance),
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
        return;
    }

    // === Only proceed with reconnection if dropping caused disconnection ===
    if n_comps <= 1 && !highest.is_empty() {
        // Single connected component after dropping — no reconnection needed
        return;
    }

    // Need to reconnect. Find nodes not on C edges.
    let highest_geoms: Vec<LineString<f64>> = highest
        .iter()
        .map(|&i| geometries[i].clone())
        .collect();

    let mut nodes_on_c = HashSet::new();
    for node in &relevant_nodes {
        let pt = Point::new(node[0], node[1]);
        for geom in &highest_geoms {
            let dist = Euclidean.distance(&pt, geom);
            if dist < params.eps {
                nodes_on_c.insert(coord_key(node));
                break;
            }
        }
    }

    let remaining_nodes: Vec<[f64; 2]> = relevant_nodes
        .iter()
        .filter(|n| !nodes_on_c.contains(&coord_key(n)))
        .copied()
        .collect();

    if remaining_nodes.is_empty() {
        return;
    }

    // === BRANCH: Multiple C edges → skeleton with filter/reconnect chain ===
    // Mirrors Python BRANCH 1 in nx_gx: skeleton → filter_connections →
    // reconnect → remove_dangles.
    if highest.len() > 1 {
        // Compute primes: shared boundary points between C edges
        let primes = compute_c_primes(&highest_geoms);

        // Compute conts_groups: C edges dissolved by connected component
        let conts_groups = dissolve_by_components(&highest_geoms);

        // Compute node degrees from full network for relevant_targets
        let mut degree_map: HashMap<(i64, i64), usize> = HashMap::new();
        for geom in geometries.iter() {
            let coords = &geom.0;
            if coords.len() < 2 {
                continue;
            }
            let c0 = coords[0];
            *degree_map.entry(coord_key(&[c0.x, c0.y])).or_default() += 1;
            let cn = coords[coords.len() - 1];
            *degree_map.entry(coord_key(&[cn.x, cn.y])).or_default() += 1;
        }

        // Relevant targets: nodes on C edges with degree > 3
        let mut target_nodes: Vec<[f64; 2]> = Vec::new();
        for node in &relevant_nodes {
            let key = coord_key(node);
            let is_on_c = {
                let pt = Point::new(node[0], node[1]);
                highest_geoms
                    .iter()
                    .any(|g| Euclidean.distance(&pt, g) < params.eps)
            };
            if is_on_c && degree_map.get(&key).copied().unwrap_or(0) > 3 {
                target_nodes.push(*node);
            }
        }
        let snap_targets = node_coords_to_lines(&target_nodes);

        // Use ALL covered edges for skeleton (matching Python)
        let all_covered_geoms: Vec<LineString<f64>> = covered_edges
            .iter()
            .map(|&i| geometries[i].clone())
            .collect();

        let snap_to = if snap_targets.is_empty() {
            None
        } else {
            Some(snap_targets.as_slice())
        };

        let (mut new_connections, _) = geometry::voronoi_skeleton(
            &all_covered_geoms,
            Some(artifact),
            snap_to,
            params.max_segment_length,
            None,
            None,
            params.clip_limit,
            Some(params.consolidation_tolerance),
        );

        // If skeleton is disconnected, retry with tiny clip_limit
        // (Python: "limit_distance was too drastic and clipped the skeleton in pieces")
        if new_connections.len() > 1 {
            let skel_comps = nodes::get_components(&new_connections);
            let n_skel_comps = skel_comps.iter().collect::<HashSet<_>>().len();
            if n_skel_comps > 1 {
                let (retry, _) = geometry::voronoi_skeleton(
                    &all_covered_geoms,
                    Some(artifact),
                    snap_to,
                    params.max_segment_length,
                    None,
                    None,
                    params.eps,
                    Some(params.consolidation_tolerance),
                );
                if !retry.is_empty() {
                    new_connections = retry;
                }
            }
        }


        new_connections =
            filter_connections(&primes, &snap_targets, &conts_groups, &new_connections);

        // Reconnect disconnected C groups
        new_connections =
            reconnect_c_groups(&conts_groups, &new_connections, artifact, params.eps);

        // Remove dangles
        let cleaned = remove_dangles(&new_connections, artifact, params.eps);
        to_add.extend(cleaned);
        return;
    }

    // === BRANCH: Single C edge → connect remaining nodes ===
    // Try shortest lines first; fall back to skeleton
    if remaining_nodes.len() == 1 {
        // One remaining node: connect to nearest C edge via shortest line
        if let Some(line) = make_shortest_to_edges(&remaining_nodes[0], &highest_geoms) {
            if geometry::is_within(&line, artifact, params.eps) {
                to_add.push(line);
                return;
            }
        }
        // Fall back to skeleton
        let es_geoms: Vec<LineString<f64>> = es_mask
            .iter()
            .map(|&i| geometries[i].clone())
            .collect();
        let snap_targets = node_coords_to_lines(&remaining_nodes);
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            Some(&highest_geoms),
            params.clip_limit,
            Some(params.consolidation_tolerance),
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
    } else {
        // Multiple remaining nodes: skeleton with C as secondary snap
        let es_geoms: Vec<LineString<f64>> = es_mask
            .iter()
            .map(|&i| geometries[i].clone())
            .collect();
        let snap_targets = node_coords_to_lines(&remaining_nodes);
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            Some(&highest_geoms),
            params.clip_limit,
            Some(params.consolidation_tolerance),
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
    }
}

/// Classify covered edges into C (continuing), E (ending), S (single) groups.
///
/// Returns (c_edges, e_edges, s_edges, es_mask, highest):
/// - c_edges: edges in C groups
/// - e_edges: edges in E groups
/// - s_edges: edges in S groups
/// - es_mask: union of E + S edges (to drop)
/// - highest: C edges (to keep)
fn classify_ces_edges(
    covered_edges: &[usize],
    coins_result: &continuity::CoinsResult,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    // Find end edges
    let end_edges: HashSet<usize> = covered_edges
        .iter()
        .filter(|&&i| coins_result.is_end[i])
        .copied()
        .collect();
    let end_groups: HashSet<usize> = end_edges.iter().map(|&i| coins_result.group[i]).collect();

    // S groups: end groups where ALL edges of the group are inside the artifact
    let mut s_groups = HashSet::new();
    for &group in &end_groups {
        let total_in_group = covered_edges
            .iter()
            .find(|&&i| coins_result.group[i] == group)
            .map(|&i| coins_result.stroke_count[i])
            .unwrap_or(0);
        let count_inside = covered_edges
            .iter()
            .filter(|&&i| coins_result.group[i] == group)
            .count();
        if count_inside == total_in_group {
            s_groups.insert(group);
        }
    }

    // E groups: end groups that aren't S
    let e_groups: HashSet<usize> = end_groups.difference(&s_groups).copied().collect();

    // C groups: covered groups that aren't end groups
    let covered_groups: HashSet<usize> =
        covered_edges.iter().map(|&i| coins_result.group[i]).collect();
    let _c_groups: HashSet<usize> = covered_groups.difference(&end_groups).copied().collect();

    let mut c_edges = Vec::new();
    let mut e_edges = Vec::new();
    let mut s_edges = Vec::new();
    let mut es_mask = Vec::new();
    let mut highest = Vec::new();

    for &i in covered_edges {
        let g = coins_result.group[i];
        if e_groups.contains(&g) {
            e_edges.push(i);
            es_mask.push(i);
        } else if s_groups.contains(&g) {
            s_edges.push(i);
            es_mask.push(i);
        } else {
            c_edges.push(i);
            highest.push(i);
        }
    }

    (c_edges, e_edges, s_edges, es_mask, highest)
}

/// Make a shortest line from a point to the nearest of a set of edges.
fn make_shortest_to_edges(point: &[f64; 2], edges: &[LineString<f64>]) -> Option<LineString<f64>> {
    let mut best_line: Option<LineString<f64>> = None;
    let mut best_dist = f64::INFINITY;

    for geom in edges {
        if let Some((pa, pb)) = ops::nearest_points(
            &LineString::new(vec![Coord { x: point[0], y: point[1] }, Coord { x: point[0], y: point[1] }]),
            geom,
        ) {
            let line = LineString::new(vec![pa, pb]);
            let len = Euclidean.length(&line);
            if len < best_dist {
                best_dist = len;
                best_line = Some(line);
            }
        }
    }
    best_line
}

/// Convert a coordinate to a hash key for deduplication.
fn coord_key(c: &[f64; 2]) -> (i64, i64) {
    ((c[0] * 1e8) as i64, (c[1] * 1e8) as i64)
}

// ─── Multi-C branch helpers ──────────────────────────────────────────────

/// Compute "primes" — boundary points shared between multiple C edges.
/// These are junction points within the C edge network.
///
/// Mirrors Python: `bd_points = highest_hierarchy.boundary.explode();
/// primes = bd_points[bd_points.duplicated()]`
fn compute_c_primes(c_geoms: &[LineString<f64>]) -> Vec<[f64; 2]> {
    let mut endpoint_counts: HashMap<(i64, i64), usize> = HashMap::new();
    let mut endpoint_coords: HashMap<(i64, i64), [f64; 2]> = HashMap::new();

    for geom in c_geoms {
        let coords = &geom.0;
        if coords.len() < 2 {
            continue;
        }

        // Start point
        let c0 = coords[0];
        let key = coord_key(&[c0.x, c0.y]);
        *endpoint_counts.entry(key).or_default() += 1;
        endpoint_coords.entry(key).or_insert([c0.x, c0.y]);

        // End point
        let cn = coords[coords.len() - 1];
        let key = coord_key(&[cn.x, cn.y]);
        *endpoint_counts.entry(key).or_default() += 1;
        endpoint_coords.entry(key).or_insert([cn.x, cn.y]);
    }

    endpoint_counts
        .iter()
        .filter(|&(_, &count)| count > 1)
        .filter_map(|(key, _)| endpoint_coords.get(key).copied())
        .collect()
}

/// Dissolve geometries by connected component labels.
/// Returns one merged (line_merge) geometry per component.
fn dissolve_by_components(geoms: &[LineString<f64>]) -> Vec<LineString<f64>> {
    if geoms.is_empty() {
        return vec![];
    }

    let labels = nodes::get_components(geoms);
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &label) in labels.iter().enumerate() {
        groups.entry(label).or_default().push(i);
    }

    groups
        .values()
        .flat_map(|indices| {
            if indices.len() == 1 {
                vec![geoms[indices[0]].clone()]
            } else {
                let group_geoms: Vec<LineString<f64>> =
                    indices.iter().map(|&i| geoms[i].clone()).collect();
                ops::line_merge(&group_geoms)
            }
        })
        .collect()
}

/// Create a shortest line between two LineString geometries.
fn make_shortest_line_between(a: &LineString<f64>, b: &LineString<f64>) -> Option<LineString<f64>> {
    let (pa, pb) = ops::nearest_points(a, b)?;
    let dx = pb.x - pa.x;
    let dy = pb.y - pa.y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 1e-10 {
        return None;
    }
    Some(LineString::new(vec![pa, pb]))
}

/// Filter skeleton connections: when multiple connections hit the same C group,
/// keep only the shortest one that reaches a target node.
///
/// Mirrors Python `filter_connections()`.
fn filter_connections(
    primes: &[[f64; 2]],
    snap_targets: &[LineString<f64>],
    conts_groups: &[LineString<f64>],
    connections: &[LineString<f64>],
) -> Vec<LineString<f64>> {
    if connections.is_empty() || conts_groups.is_empty() {
        return connections.to_vec();
    }

    // Build union of all target points as small lines for intersection testing
    let mut all_target_pts: Vec<Point<f64>> = Vec::new();
    for p in primes {
        all_target_pts.push(Point::new(p[0], p[1]));
    }
    for snap in snap_targets {
        if !snap.0.is_empty() {
            all_target_pts.push(Point::new(snap.0[0].x, snap.0[0].y));
        }
    }

    let mut unwanted: HashSet<usize> = HashSet::new();
    let mut keeping: Vec<LineString<f64>> = Vec::new();

    for c_group in conts_groups {
        // Find connections that intersect this C group
        let intersecting_c: Vec<usize> = connections
            .iter()
            .enumerate()
            .filter(|(_, conn)| conn.intersects(c_group))
            .map(|(i, _)| i)
            .collect();

        if intersecting_c.len() > 1 {
            // Multiple connections to this C — find which ones reach targets
            let reaching_targets: Vec<usize> = if !all_target_pts.is_empty() {
                intersecting_c
                    .iter()
                    .filter(|&&i| {
                        all_target_pts.iter().any(|pt| {
                            Euclidean.distance(pt, &connections[i]) < 5.0
                        })
                    })
                    .copied()
                    .collect()
            } else {
                vec![]
            };

            if !reaching_targets.is_empty() {
                // Keep only shortest connection that reaches a target
                let shortest_idx = reaching_targets
                    .iter()
                    .min_by(|&&a, &&b| {
                        let la = Euclidean.length(&connections[a]);
                        let lb = Euclidean.length(&connections[b]);
                        la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied();

                // Mark all intersecting-C as unwanted
                for &i in &intersecting_c {
                    unwanted.insert(i);
                }
                // Add back the shortest
                if let Some(idx) = shortest_idx {
                    keeping.push(connections[idx].clone());
                }
            } else {
                // Fork case: multiple C connections, none reaching targets
                // Mark all as unwanted — reconnect will recover if needed
                for &i in &intersecting_c {
                    unwanted.insert(i);
                }
            }
        }
    }

    let mut result: Vec<LineString<f64>> = connections
        .iter()
        .enumerate()
        .filter(|(i, _)| !unwanted.contains(i))
        .map(|(_, g)| g.clone())
        .collect();
    result.extend(keeping);
    result
}

/// Check for disconnected C groups and reconnect via shortest lines.
///
/// Mirrors Python `reconnect()`.
fn reconnect_c_groups(
    conts_groups: &[LineString<f64>],
    connections: &[LineString<f64>],
    artifact: &Polygon<f64>,
    eps: f64,
) -> Vec<LineString<f64>> {
    if connections.is_empty() || conts_groups.is_empty() {
        return connections.to_vec();
    }

    // Dissolve connections by connected component
    let conn_dissolved = dissolve_by_components(connections);

    let mut additions = Vec::new();
    for c_group in conts_groups {
        let all_intersect = conn_dissolved.iter().all(|comp| {
            Euclidean.distance(comp, c_group) <= eps
        });

        if !all_intersect {
            // Some components don't reach this C — add shortest connections
            for comp in &conn_dissolved {
                let intersects = Euclidean.distance(comp, c_group) <= eps;

                if !intersects {
                    if let Some(sl) = make_shortest_line_between(comp, c_group) {
                        if geometry::is_within(&sl, artifact, eps) {
                            additions.push(sl);
                        }
                    }
                }
            }
        }
    }

    let mut result = connections.to_vec();
    result.extend(additions);
    result
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Apply accumulated drops and additions to the network.
///
/// Post-processing matches Python:
/// 1. Drop marked edges
/// 2. Merge new additions via line_merge → explode to single LineStrings
/// 3. Deduplicate
/// 4. Simplify new edges by max_segment_length * simplification_factor
/// 5. Clean topology via remove_interstitial_nodes
fn apply_changes(
    network: &mut StreetNetwork,
    to_drop: &[usize],
    to_add: &[LineString<f64>],
    params: &NeatifyParams,
) {
    if to_drop.is_empty() && to_add.is_empty() {
        return;
    }

    let drop_set: HashSet<usize> = to_drop.iter().copied().collect();

    // Compute union of parent_ids from ALL dropped edges BEFORE reassignment
    let mut dropped_parents: Vec<usize> = Vec::new();
    for &i in to_drop {
        if i < network.parent_ids.len() {
            dropped_parents.extend_from_slice(&network.parent_ids[i]);
        }
    }
    dropped_parents.sort_unstable();
    dropped_parents.dedup();

    // Remove dropped edges in-place using swap_remove pattern to avoid
    // cloning all surviving geometries. We collect indices to remove in
    // reverse order so swap_remove doesn't invalidate earlier indices.
    let mut sorted_drops: Vec<usize> = drop_set.iter().copied().collect();
    sorted_drops.sort_unstable();
    for &i in sorted_drops.iter().rev() {
        if i < network.geometries.len() {
            network.geometries.swap_remove(i);
            network.statuses.swap_remove(i);
            network.parent_ids.swap_remove(i);
        }
    }

    // Post-process additions: line_merge, explode, dedup, simplify
    if !to_add.is_empty() {
        let merged_adds = merge_and_explode(to_add);
        let deduped = dedup_geometries(&merged_adds);
        drop(merged_adds);
        let simp_eps = params.max_segment_length * params.simplification_factor;
        let valid_edges: Vec<LineString<f64>> = deduped
            .into_iter()
            .filter_map(|geom| {
                let simplified = geom.simplify(simp_eps);
                if simplified.0.len() >= 2 && Euclidean.length(&simplified) > params.eps {
                    Some(simplified)
                } else {
                    None
                }
            })
            .collect();
        let n_valid = valid_edges.len();
        for (j, simplified) in valid_edges.into_iter().enumerate() {
            network.geometries.push(simplified);
            network.statuses.push(EdgeStatus::New);
            // Move dropped_parents into the last edge; others get empty.
            // Avoids cloning a potentially huge vector (e.g. 72K entries)
            // for every new edge, which can use multiple GB.
            if j + 1 == n_valid {
                network.parent_ids.push(dropped_parents);
                break;
            } else {
                network.parent_ids.push(vec![]);
            }
        }
    }

    // Clean topology after changes
    let (cleaned, clean_statuses, cleaned_parents) =
        nodes::remove_interstitial_nodes(
            std::mem::take(&mut network.geometries),
            std::mem::take(&mut network.statuses),
            std::mem::take(&mut network.parent_ids),
        );
    network.geometries = cleaned;
    network.statuses = clean_statuses;
    network.parent_ids = cleaned_parents;
}

/// Check if a CES classification represents an "identical" case
/// (all strokes are the same type: all C, all E, or all S).
fn is_identical_ces(ces: &continuity::CesInfo) -> bool {
    let types_present =
        (if ces.c > 0 { 1 } else { 0 })
        + (if ces.e > 0 { 1 } else { 0 })
        + (if ces.s > 0 { 1 } else { 0 });
    types_present <= 1
}

/// Merge a set of geometries via line_merge, then explode into individual LineStrings.
fn merge_and_explode(geoms: &[LineString<f64>]) -> Vec<LineString<f64>> {
    if geoms.is_empty() {
        return vec![];
    }
    let merged = ops::line_merge(geoms);
    if merged.is_empty() {
        return geoms.to_vec();
    }
    merged
}

/// Deduplicate geometries by normalized coordinate hash.
fn dedup_geometries(geoms: &[LineString<f64>]) -> Vec<LineString<f64>> {
    use std::hash::{Hash, Hasher};
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for geom in geoms {
        let normalized = ops::normalize_linestring(geom);
        // Hash coordinates directly instead of serializing to WKT string
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        normalized.0.len().hash(&mut hasher);
        for c in &normalized.0 {
            c.x.to_bits().hash(&mut hasher);
            c.y.to_bits().hash(&mut hasher);
        }
        if seen.insert(hasher.finish()) {
            result.push(geom.clone());
        }
    }
    result
}

/// Deduplicate a StreetNetwork in-place by normalized coordinate hash.
/// Duplicate edges union their parent_ids into the survivor.
fn dedup_network(network: &mut StreetNetwork) {
    use std::hash::{Hash, Hasher};
    let n = network.geometries.len();

    // Phase 1: Hash all geometries and identify duplicates in-place.
    // Maps hash → first occurrence index. Duplicates get their parent_ids
    // merged into the survivor and are marked for removal.
    let mut seen: HashMap<u64, usize> = HashMap::with_capacity(n);
    let mut to_remove = Vec::new();

    for i in 0..n {
        let normalized = ops::normalize_linestring(&network.geometries[i]);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        normalized.0.len().hash(&mut hasher);
        for c in &normalized.0 {
            c.x.to_bits().hash(&mut hasher);
            c.y.to_bits().hash(&mut hasher);
        }
        let hash = hasher.finish();
        if let Some(&survivor_idx) = seen.get(&hash) {
            // Merge parent_ids into the survivor. Use a temporary to avoid
            // double-borrowing network.parent_ids.
            let dup_parents = std::mem::take(&mut network.parent_ids[i]);
            let survivor_parents = &mut network.parent_ids[survivor_idx];
            for p in dup_parents {
                if !survivor_parents.contains(&p) {
                    survivor_parents.push(p);
                }
            }
            survivor_parents.sort_unstable();
            to_remove.push(i);
        } else {
            seen.insert(hash, i);
        }
    }

    if to_remove.is_empty() {
        return;
    }

    // Phase 2: Remove duplicates by compacting in-place (reverse order
    // swap_remove to avoid index invalidation).
    to_remove.sort_unstable();
    for &i in to_remove.iter().rev() {
        network.geometries.swap_remove(i);
        network.statuses.swap_remove(i);
        network.parent_ids.swap_remove(i);
    }
}

/// Find edge indices whose geometry is covered by the artifact polygon.
#[cfg(test)]
fn find_covered_edges(
    geometries: &[LineString<f64>],
    artifact: &Polygon<f64>,
    eps: f64,
) -> Vec<usize> {
    let tree = crate::spatial::build_rtree(geometries);
    find_covered_edges_with_tree(geometries, &tree, artifact, eps)
}

/// find_covered_edges using a pre-built R-tree.
///
/// Uses distance-based coverage checks instead of buffer+containment to avoid
/// expensive polygon construction from buffer(). Semantically equivalent to
/// `artifact.buffer(eps).covers(line)`.
fn find_covered_edges_with_tree(
    geometries: &[LineString<f64>],
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    artifact: &Polygon<f64>,
    eps: f64,
) -> Vec<usize> {
    // Expand the R-tree query envelope by eps to catch edges within distance
    let candidates = nodes::envelope_query_indices_expanded(tree, artifact, eps);

    // Use rayon for parallel distance checks.
    use rayon::prelude::*;
    let covered: Vec<bool> = candidates
        .par_iter()
        .map(|&i| {
            line_covered_by_distance(&geometries[i], artifact, eps)
        })
        .collect();
    candidates
        .into_iter()
        .zip(covered)
        .filter(|&(_, is_covered)| is_covered)
        .map(|(i, _)| i)
        .collect()
}

/// Find edges that intersect the artifact boundary but are not fully covered.
#[cfg(test)]
fn find_boundary_edges(
    geometries: &[LineString<f64>],
    artifact: &Polygon<f64>,
    eps: f64,
) -> Vec<usize> {
    let tree = crate::spatial::build_rtree(geometries);
    find_boundary_edges_with_tree(geometries, &tree, artifact, eps)
}

/// find_boundary_edges using a pre-built R-tree.
///
/// Uses distance-based checks instead of buffer+intersects to avoid expensive
/// polygon construction. An edge is a boundary edge if:
/// - It is NOT fully covered (distance-based), AND
/// - It is within eps of the artifact exterior ring
fn find_boundary_edges_with_tree(
    geometries: &[LineString<f64>],
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    artifact: &Polygon<f64>,
    eps: f64,
) -> Vec<usize> {
    // Expand the R-tree query envelope by eps to catch nearby edges
    let candidates = nodes::envelope_query_indices_expanded(tree, artifact, eps);

    let exterior = artifact.exterior();

    // Use rayon for parallel distance checks.
    use rayon::prelude::*;
    let results: Vec<bool> = candidates
        .par_iter()
        .map(|&i| {
            let geom = &geometries[i];
            let is_covered = line_covered_by_distance(geom, artifact, eps);
            if is_covered {
                return false;
            }
            // Check if edge is within eps of the artifact exterior boundary
            Euclidean.distance(geom, exterior) <= eps
        })
        .collect();
    candidates
        .into_iter()
        .zip(results)
        .filter(|&(_, keep)| keep)
        .map(|(i, _)| i)
        .collect()
}

/// Find where boundary-crossing edges intersect the polygon exterior ring.
///
/// Computes actual line-segment intersection points between boundary edge
/// segments and exterior ring segments. These crossing points are where the
/// polygon connects to the broader network, used to split the ring into "sides".
fn find_connection_points(
    geometries: &[LineString<f64>],
    boundary_edges: &[usize],
    exterior: &LineString<f64>,
    _tol: f64,
) -> Vec<Coord<f64>> {
    let mut points = Vec::new();
    for &i in boundary_edges {
        let edge = &geometries[i];
        for edge_seg in edge.lines() {
            for ring_seg in exterior.lines() {
                if let Some(pt) = segment_intersection(&edge_seg, &ring_seg) {
                    points.push(pt);
                }
            }
        }
    }
    points
}

/// Compute the intersection point of two line segments, if any.
fn segment_intersection(a: &Line<f64>, b: &Line<f64>) -> Option<Coord<f64>> {
    let x1 = a.start.x; let y1 = a.start.y;
    let x2 = a.end.x;   let y2 = a.end.y;
    let x3 = b.start.x; let y3 = b.start.y;
    let x4 = b.end.x;   let y4 = b.end.y;

    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denom.abs() < 1e-12 {
        return None; // Parallel or coincident
    }

    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    let u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

    if t >= -1e-9 && t <= 1.0 + 1e-9 && u >= -1e-9 && u <= 1.0 + 1e-9 {
        Some(Coord {
            x: x1 + t * (x2 - x1),
            y: y1 + t * (y2 - y1),
        })
    } else {
        None
    }
}

/// Split a polygon exterior ring at given points, returning the arcs between split points.
///
/// For each split point, finds the nearest ring vertex and splits there. This handles
/// crossing points that fall between ring vertices (from segment_intersection).
/// Returns arcs between consecutive split positions. Needs at least 2 distinct split
/// positions to produce 2+ arcs.
fn split_ring_at_points(
    ring: &LineString<f64>,
    split_points: &[Coord<f64>],
    _tol: f64,
) -> Vec<LineString<f64>> {
    if split_points.is_empty() || ring.0.len() < 4 {
        return vec![ring.clone()];
    }

    let n = ring.0.len() - 1; // Exclude closing duplicate vertex

    // For each split point, find the nearest ring vertex index
    let mut split_indices: Vec<usize> = Vec::new();
    for sp in split_points {
        let mut best_idx = 0;
        let mut best_dist_sq = f64::INFINITY;
        for (i, coord) in ring.0[..n].iter().enumerate() {
            let dx = coord.x - sp.x;
            let dy = coord.y - sp.y;
            let d_sq = dx * dx + dy * dy;
            if d_sq < best_dist_sq {
                best_dist_sq = d_sq;
                best_idx = i;
            }
        }
        split_indices.push(best_idx);
    }

    // Deduplicate and sort
    split_indices.sort_unstable();
    split_indices.dedup();

    if split_indices.len() < 2 {
        // Need at least 2 split positions to create 2+ arcs
        return vec![ring.clone()];
    }

    // Walk the ring from each split index to the next, collecting arcs
    let mut arcs: Vec<LineString<f64>> = Vec::new();

    for w in 0..split_indices.len() {
        let start = split_indices[w];
        let end = split_indices[(w + 1) % split_indices.len()];

        let mut coords: Vec<Coord<f64>> = Vec::new();
        let mut i = start;
        loop {
            coords.push(ring.0[i]);
            if i == end && coords.len() > 1 {
                break;
            }
            i = (i + 1) % n;
            if coords.len() > n + 1 {
                break; // Safety: prevent infinite loop
            }
        }

        if coords.len() >= 2 {
            arcs.push(LineString::new(coords));
        }
    }

    arcs.retain(|ls| ls.0.len() >= 2 && Euclidean.length(ls) > 0.0);
    arcs
}

/// Find network nodes near a polygon (within eps).
fn find_nodes_near_polygon(
    node_coords: &[[f64; 2]],
    polygon: &Polygon<f64>,
    eps: f64,
) -> Vec<[f64; 2]> {
    let mut result = Vec::new();
    for coord in node_coords {
        let pt = Point::new(coord[0], coord[1]);
        let dist = Euclidean.distance(&pt, polygon);
        if dist <= eps {
            result.push(*coord);
        }
    }
    result
}

/// Remove dangling edges from skeleton output.
///
/// After line_merge, find endpoints that have no connection to any other
/// geometry or the artifact boundary, and remove those edges.
/// Uses endpoint-to-geometry distance (not just endpoint-to-endpoint) to
/// handle T-junctions that Rust's line_merge doesn't split precisely.
///
/// Mirrors Python `remove_dangles()` (artifacts.py:672-708).
fn remove_dangles(connections: &[LineString<f64>], artifact: &Polygon<f64>, eps: f64) -> Vec<LineString<f64>> {
    if connections.len() <= 1 {
        return connections.to_vec();
    }

    // Line merge first, then explode
    let merged = merge_and_explode(connections);
    if merged.len() <= 1 {
        return merged;
    }

    let boundary = artifact.exterior().clone();
    // Python uses eps=1e-4 with GEOS line_merge (perfect junctions).
    // Rust's line_merge is less precise, so use a generous tolerance.
    // Check endpoint-to-full-geometry distance to catch T-junctions where
    // an endpoint lands near the middle of another edge.
    let snap_tol = eps.max(1.0);

    let mut dangle_edges: HashSet<usize> = HashSet::new();

    for (i, line) in merged.iter().enumerate() {
        let coords = &line.0;
        if coords.len() < 2 {
            continue;
        }

        for endpoint in [coords[0], coords[coords.len() - 1]] {
            let pt = Point::new(endpoint.x, endpoint.y);
            let mut connected = false;

            // Check endpoint distance to other geometries (full geometry, not just endpoints)
            for (j, other) in merged.iter().enumerate() {
                if i == j {
                    continue;
                }
                if Euclidean.distance(&pt, other) < snap_tol {
                    connected = true;
                    break;
                }
            }

            // Check against artifact boundary
            if !connected && Euclidean.distance(&pt, &boundary) < snap_tol {
                connected = true;
            }

            if !connected {
                dangle_edges.insert(i);
            }
        }
    }

    if dangle_edges.is_empty() {
        return merged;
    }

    merged
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !dangle_edges.contains(i))
        .map(|(_, g)| g)
        .collect()
}

/// Convert node coordinate arrays to small LineString geometries for snap targets.
///
/// Creates 2-point degenerate LineStrings (point-like) that the voronoi_skeleton
/// snap_to parameter expects.
fn node_coords_to_lines(coords: &[[f64; 2]]) -> Vec<LineString<f64>> {
    coords
        .iter()
        .map(|c| {
            LineString::new(vec![
                Coord { x: c[0], y: c[1] },
                Coord { x: c[0], y: c[1] },
            ])
        })
        .collect()
}

/// Fast containment check: avoids full `Relate` when all vertices + midpoints are inside.
///
/// If every vertex and segment midpoint of `line` is inside `poly`, the line is covered.
/// Falls back to full `poly.relate(line).is_covers()` when any test point is outside.
/// Check if a line is "covered" by an artifact polygon within eps distance.
///
/// Semantically equivalent to `buffer(eps).covers(line)` but uses distance
/// checks instead, avoiding the expensive polygon construction from buffer().
/// For each vertex and each segment midpoint of the line, checks that the
/// distance to the artifact is <= eps.
fn line_covered_by_distance(line: &LineString<f64>, artifact: &Polygon<f64>, eps: f64) -> bool {
    for coord in &line.0 {
        let pt = Point::new(coord.x, coord.y);
        if Euclidean.distance(&pt, artifact) > eps {
            return false;
        }
    }
    for w in line.0.windows(2) {
        let mid = Point::new((w[0].x + w[1].x) / 2.0, (w[0].y + w[1].y) / 2.0);
        if Euclidean.distance(&mid, artifact) > eps {
            return false;
        }
    }
    true
}

fn line_covered_by_polygon_fast(line: &LineString<f64>, poly: &Polygon<f64>) -> bool {
    for coord in &line.0 {
        let pt = Point::new(coord.x, coord.y);
        if !poly.contains(&pt) {
            return poly.relate(line).is_covers();
        }
    }
    for w in line.0.windows(2) {
        let mid = Point::new((w[0].x + w[1].x) / 2.0, (w[0].y + w[1].y) / 2.0);
        if !poly.contains(&mid) {
            return poly.relate(line).is_covers();
        }
    }
    true
}

/// Error type for the neatify pipeline.
#[derive(Debug, thiserror::Error)]
pub enum NeatifyError {
    #[error("Geometry error: {0}")]
    Geometry(String),
    #[error("No projected CRS set on input data")]
    NoCrs,
    #[error("Artifact detection failed: {0}")]
    ArtifactDetection(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(coords: &[[f64; 2]]) -> LineString<f64> {
        LineString::new(
            coords
                .iter()
                .map(|c| Coord { x: c[0], y: c[1] })
                .collect(),
        )
    }

    fn make_poly(coords: &[[f64; 2]]) -> Polygon<f64> {
        Polygon::new(
            LineString::new(
                coords
                    .iter()
                    .map(|c| Coord { x: c[0], y: c[1] })
                    .collect(),
            ),
            vec![],
        )
    }

    #[test]
    fn test_neatify_params_default() {
        let params = NeatifyParams::default();
        assert_eq!(params.n_loops, 2);
        assert_eq!(params.angle_threshold, 120.0);
    }

    #[test]
    fn test_is_identical_ces() {
        // All C
        assert!(is_identical_ces(&continuity::CesInfo {
            stroke_count: 2, c: 2, e: 0, s: 0,
        }));
        // All E
        assert!(is_identical_ces(&continuity::CesInfo {
            stroke_count: 3, c: 0, e: 3, s: 0,
        }));
        // Mixed C+E
        assert!(!is_identical_ces(&continuity::CesInfo {
            stroke_count: 3, c: 1, e: 2, s: 0,
        }));
        // Empty (no strokes)
        assert!(is_identical_ces(&continuity::CesInfo {
            stroke_count: 0, c: 0, e: 0, s: 0,
        }));
    }

    #[test]
    fn test_classify_ces_edges() {
        // Build a small coins result with known C/E/S classification
        // 4 edges: edges 0,1 in group 0 (both ends), edge 2 in group 1 (not end),
        //          edge 3 in group 2 (end, only member)
        let coins = continuity::CoinsResult {
            group: vec![0, 0, 1, 2],
            is_end: vec![true, true, false, true],
            stroke_length: vec![10.0, 10.0, 15.0, 5.0],
            stroke_count: vec![2, 2, 1, 1],
            n_segments: 4,
            n_p1_confirmed: 0,
            n_p2_confirmed: 0,
        };

        let covered = vec![0, 1, 2, 3];
        let (c_edges, e_edges, s_edges, es_mask, highest) =
            classify_ces_edges(&covered, &coins);

        // Group 0: both edges are end, both are inside → S (2 of 2)
        // Group 1: edge 2 not end → C
        // Group 2: edge 3 end, 1 of 1 inside → S
        assert!(c_edges.contains(&2), "edge 2 should be C");
        assert!(s_edges.contains(&0) && s_edges.contains(&1), "edges 0,1 should be S");
        assert!(s_edges.contains(&3), "edge 3 should be S");
        assert!(e_edges.is_empty(), "no E edges in this case");
        assert_eq!(highest.len(), 1);
        assert_eq!(es_mask.len(), 3);
    }

    #[test]
    fn test_find_covered_edges() {
        let poly = make_poly(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]);
        let inside = make_line(&[[2.0, 5.0], [8.0, 5.0]]);
        let outside = make_line(&[[12.0, 5.0], [18.0, 5.0]]);
        let crossing = make_line(&[[5.0, 5.0], [15.0, 5.0]]);

        let geoms = vec![inside, outside, crossing];
        let covered = find_covered_edges(&geoms, &poly, 0.001);

        assert!(covered.contains(&0), "inside line should be covered");
        assert!(!covered.contains(&1), "outside line should not be covered");
        assert!(!covered.contains(&2), "crossing line should not be covered");
    }

    #[test]
    fn test_find_boundary_edges() {
        let poly = make_poly(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]);
        let inside = make_line(&[[2.0, 5.0], [8.0, 5.0]]);
        let boundary = make_line(&[[5.0, -5.0], [5.0, 5.0]]);
        let outside = make_line(&[[12.0, 5.0], [18.0, 5.0]]);

        let geoms = vec![inside, boundary, outside];
        let boundary_edges = find_boundary_edges(&geoms, &poly, 0.001);

        assert!(!boundary_edges.contains(&0), "inside line is not a boundary edge");
        assert!(boundary_edges.contains(&1), "crossing line is a boundary edge");
        assert!(!boundary_edges.contains(&2), "outside line is not a boundary edge");
    }

    #[test]
    fn test_merge_and_explode() {
        // Two connected linestrings should merge into one
        let l1 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let l2 = make_line(&[[5.0, 0.0], [10.0, 0.0]]);
        let result = merge_and_explode(&[l1, l2]);
        assert_eq!(result.len(), 1, "two connected lines should merge to one");

        // Two disconnected linestrings should stay as two
        let l3 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let l4 = make_line(&[[20.0, 0.0], [25.0, 0.0]]);
        let result2 = merge_and_explode(&[l3, l4]);
        assert_eq!(result2.len(), 2, "two disconnected lines should stay as two");
    }

    #[test]
    fn test_dedup_geometries() {
        let l1 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let l2 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let l3 = make_line(&[[10.0, 0.0], [15.0, 0.0]]);
        let result = dedup_geometries(&[l1, l2, l3]);
        assert_eq!(result.len(), 2, "duplicate should be removed");
    }

    #[test]
    fn test_coord_key() {
        let c1 = [1.23456789, 4.56789012];
        let c2 = [1.23456789, 4.56789012];
        let c3 = [1.234567891, 4.56789012]; // differs at 10th decimal
        assert_eq!(coord_key(&c1), coord_key(&c2));
        // c3 might or might not match depending on floating point precision
        // but should be stable for same input
    }

    #[test]
    fn test_make_shortest_to_edges() {
        let edge = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let point = [5.0, 3.0];
        let line = make_shortest_to_edges(&point, &[edge]).unwrap();
        let len = Euclidean.length(&line);
        assert!((len - 3.0).abs() < 0.01, "shortest line from (5,3) to x-axis should be ~3");
    }

    #[test]
    fn test_split_cluster_spatially_preserves_all_artifacts() {
        // Create 10 artifact polygons spread across space, including some that
        // will land alone in a grid cell (orphans with < 3 per cell).
        let mut geoms = Vec::new();
        for i in 0..10 {
            let x = (i as f64) * 100.0;
            geoms.push(make_poly(&[
                [x, 0.0], [x + 1.0, 0.0], [x + 1.0, 1.0], [x, 1.0], [x, 0.0],
            ]));
        }

        let cluster: Vec<usize> = (0..10).collect();
        let result = split_cluster_spatially(&cluster, &geoms, 3);

        // Collect all artifact indices from the result
        let mut all_indices: Vec<usize> = result.into_iter().flatten().collect();
        all_indices.sort();

        assert_eq!(
            all_indices,
            (0..10).collect::<Vec<_>>(),
            "all artifact indices must be preserved — none should be dropped"
        );
    }

    #[test]
    fn test_split_cluster_spatially_all_tiny_returns_original() {
        // When all sub-clusters would be < 3, return original cluster unsplit
        let geoms: Vec<Polygon<f64>> = (0..4)
            .map(|i| {
                let x = (i as f64) * 1000.0; // Far apart — each in its own cell
                make_poly(&[
                    [x, 0.0], [x + 1.0, 0.0], [x + 1.0, 1.0], [x, 1.0], [x, 0.0],
                ])
            })
            .collect();

        let cluster: Vec<usize> = (0..4).collect();
        let result = split_cluster_spatially(&cluster, &geoms, 2);

        // Should return a single cluster with all artifacts
        assert_eq!(result.len(), 1, "should return single unsplit cluster");
        let mut indices: Vec<usize> = result[0].clone();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
