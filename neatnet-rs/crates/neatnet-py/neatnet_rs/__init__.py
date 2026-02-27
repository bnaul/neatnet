"""neatnet-rs: Rust-accelerated street network simplification."""

import geopandas as gpd
import pandas as pd
import pyarrow as pa

from neatnet_rs._neatnet_rs import (
    coins as _coins_arrow,
    coins_wkt,
    diagnostics as _diagnostics_arrow,
    neatify as _neatify_arrow,
    neatify_wkt,
    version,
    voronoi_skeleton_wkt,
)


def neatify(streets: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
    """Simplify a street network.

    Parameters
    ----------
    streets : GeoDataFrame
        Input street network with LineString geometry.
        The DataFrame index is preserved through the pipeline: output
        ``parent_ids`` will reference the original index values.
    **kwargs
        Forwarded to the Rust neatify function.

    Returns
    -------
    GeoDataFrame
        Simplified street network with geometry, status, and parent_ids columns.
        parent_ids is a PyArrow-backed list<uint64> column for zero-copy performance.
    """
    table = streets.to_arrow(geometry_encoding="geoarrow")
    edge_ids = streets.index.to_numpy(dtype="uint64").tolist()
    arrow_table = _neatify_arrow(table, edge_ids=edge_ids, **kwargs)
    # Pull parent_ids out before GeoDataFrame.from_arrow() (which would
    # materialize it as Python lists), then re-attach as ArrowDtype-backed.
    parent_ids_col = arrow_table.column("parent_ids")
    gdf = gpd.GeoDataFrame.from_arrow(arrow_table.drop("parent_ids"))
    gdf["parent_ids"] = pd.arrays.ArrowExtensionArray(parent_ids_col.combine_chunks())
    return gdf


def coins(streets: gpd.GeoDataFrame, *, angle_threshold: float = 120.0) -> gpd.GeoDataFrame:
    """Run COINS continuity analysis.

    Parameters
    ----------
    streets : GeoDataFrame
        Input street network with LineString geometry.
    angle_threshold : float, default 120.0
        Angle threshold for continuity detection.

    Returns
    -------
    GeoDataFrame
        Input data with additional COINS columns (group, is_end,
        stroke_length, stroke_count).
    """
    table = streets.to_arrow(geometry_encoding="geoarrow")
    result = _coins_arrow(table, angle_threshold=angle_threshold)
    return gpd.GeoDataFrame.from_arrow(result)


def diagnostics(streets: gpd.GeoDataFrame, **kwargs) -> dict:
    """Run diagnostic / dry-run mode on a street network.

    Runs the cheap setup phases (fix_topology, consolidate_nodes, artifact
    detection, classification) and returns complexity metrics without
    performing any simplification.  Covers ~1-5 % of total ``neatify``
    runtime but reveals all the signals needed to estimate cost.

    Parameters
    ----------
    streets : GeoDataFrame
        Input street network with LineString geometry.
    **kwargs
        Forwarded to the Rust diagnostics function (max_segment_length,
        consolidation_tolerance, artifact_threshold,
        artifact_threshold_fallback, eps).

    Returns
    -------
    dict
        Keys: n_input_edges, n_edges_after_topology, n_artifacts,
        fai_threshold, n_singles, n_pairs, n_cluster_artifacts,
        n_cluster_groups, max_cluster_size, fix_topology_secs,
        consolidate_secs, artifact_detection_secs.
    """
    table = streets.to_arrow(geometry_encoding="geoarrow")
    return dict(_diagnostics_arrow(table, **kwargs))


__all__ = [
    "coins",
    "coins_wkt",
    "diagnostics",
    "neatify",
    "neatify_wkt",
    "version",
    "voronoi_skeleton_wkt",
]
__version__ = version()
