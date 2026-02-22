"""neatnet-rs: Rust-accelerated street network simplification."""

import geopandas as gpd
import pandas as pd
import pyarrow as pa

from neatnet_rs._neatnet_rs import (
    coins as _coins_arrow,
    coins_wkt,
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
    **kwargs
        Forwarded to the Rust neatify function.

    Returns
    -------
    GeoDataFrame
        Simplified street network with geometry, status, and parent_ids columns.
        parent_ids is a PyArrow-backed list<uint32> column for zero-copy performance.
    """
    table = streets.to_arrow(geometry_encoding="geoarrow")
    arrow_table = _neatify_arrow(table, **kwargs)
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


__all__ = [
    "coins",
    "coins_wkt",
    "neatify",
    "neatify_wkt",
    "version",
    "voronoi_skeleton_wkt",
]
__version__ = version()
