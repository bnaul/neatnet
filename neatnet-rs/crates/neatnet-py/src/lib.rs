//! PyO3 Python bindings for neatnet-rs.
//!
//! Exposes the Rust neatify pipeline to Python via Arrow tables for
//! zero-copy geometry transfer.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use geoarrow::array::{
    from_arrow_array, AsGeoArrowArray, GeoArrowArray, GeoArrowArrayAccessor, LineStringBuilder,
};
use geoarrow::datatypes::{Dimension, LineStringType, Metadata};
use geo_traits::to_geo::ToGeoLineString;
use geo_types::LineString;
use pyo3::prelude::*;
use pyo3_arrow::PyTable;

/// The neatnet_rs Python module.
#[pymodule]
fn _neatnet_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neatify, m)?)?;
    m.add_function(wrap_pyfunction!(neatify_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(coins, m)?)?;
    m.add_function(wrap_pyfunction!(coins_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_skeleton_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

/// Return the version of the neatnet-rs Rust core.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// Arrow table helpers
// ---------------------------------------------------------------------------

/// Extract LineString geometries and CRS metadata from an Arrow table.
fn table_to_linestrings(
    table: PyTable,
) -> PyResult<(Vec<LineString<f64>>, RecordBatch, Arc<Schema>, Arc<Metadata>)> {
    let (batches, schema) = table.into_inner();

    let geom_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "geometry")
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No 'geometry' column found in table")
        })?;

    let batch = concat_batches(&schema, &batches)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let ga = from_arrow_array(batch.column(geom_idx).as_ref(), schema.field(geom_idx))
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;

    let metadata = ga.data_type().metadata().clone();

    let ls_array = ga.as_line_string_opt().ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("Expected a GeoArrow LineString array")
    })?;

    let mut geoms = Vec::with_capacity(ls_array.len());
    for i in 0..ls_array.len() {
        if ls_array.is_null(i) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Null geometry at index {i}"
            )));
        }
        let scalar = ls_array.value(i).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Array access error at index {i}: {e}"
            ))
        })?;
        geoms.push(scalar.to_line_string());
    }

    Ok((geoms, batch, schema, metadata))
}

// ---------------------------------------------------------------------------
// WKT helpers
// ---------------------------------------------------------------------------

/// Parse a WKT string to a geo_types::LineString<f64>.
fn wkt_to_linestring(wkt_str: &str) -> Option<LineString<f64>> {
    use std::str::FromStr;
    let wkt_obj = wkt::Wkt::from_str(wkt_str).ok()?;
    let geom: geo_types::Geometry<f64> = wkt_obj.try_into().ok()?;
    match geom {
        geo_types::Geometry::LineString(ls) => Some(ls),
        _ => None,
    }
}

/// Convert a geo_types::LineString<f64> to a WKT string.
fn linestring_to_wkt(ls: &LineString<f64>) -> String {
    use std::fmt::Write;
    let mut s = String::from("LINESTRING (");
    for (i, coord) in ls.0.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{} {}", coord.x, coord.y).unwrap();
    }
    s.push(')');
    s
}

/// Parse a WKT string to a geo_types::Polygon<f64>.
fn wkt_to_polygon(wkt_str: &str) -> Option<geo_types::Polygon<f64>> {
    use std::str::FromStr;
    let wkt_obj = wkt::Wkt::from_str(wkt_str).ok()?;
    let geom: geo_types::Geometry<f64> = wkt_obj.try_into().ok()?;
    match geom {
        geo_types::Geometry::Polygon(p) => Some(p),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Arrow-table-based functions (primary API)
// ---------------------------------------------------------------------------

/// Simplify a street network.
///
/// Accepts an Arrow table (from ``GeoDataFrame.to_arrow()``) and returns a
/// new Arrow table with geometry and status columns.
#[pyfunction]
#[pyo3(signature = (
    table,
    *,
    edge_ids=None,
    max_segment_length=1.0,
    min_dangle_length=20.0,
    clip_limit=2.0,
    simplification_factor=2.0,
    consolidation_tolerance=10.0,
    artifact_threshold=None,
    artifact_threshold_fallback=7.0,
    angle_threshold=120.0,
    eps=1e-4,
    n_loops=2,
))]
fn neatify<'py>(
    py: Python<'py>,
    table: PyTable,
    edge_ids: Option<Vec<u64>>,
    max_segment_length: f64,
    min_dangle_length: f64,
    clip_limit: f64,
    simplification_factor: f64,
    consolidation_tolerance: f64,
    artifact_threshold: Option<f64>,
    artifact_threshold_fallback: f64,
    angle_threshold: f64,
    eps: f64,
    n_loops: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let (geo_geoms, _batch, _schema, metadata) = table_to_linestrings(table)?;
    let n_input = geo_geoms.len();
    let statuses = vec![neatnet_core::EdgeStatus::Original; n_input];
    let parent_ids: Vec<Vec<usize>> = match edge_ids {
        Some(ids) => {
            if ids.len() != n_input {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("edge_ids length ({}) != number of geometries ({})", ids.len(), n_input)
                ));
            }
            ids.into_iter().map(|id| vec![id as usize]).collect()
        }
        None => (0..n_input).map(|i| vec![i]).collect(),
    };

    let mut network = neatnet_core::StreetNetwork {
        geometries: geo_geoms,
        statuses,
        parent_ids,
        attributes: None,
        crs: None,
    };

    let params = neatnet_core::NeatifyParams {
        max_segment_length,
        min_dangle_length,
        clip_limit,
        simplification_factor,
        consolidation_tolerance,
        artifact_threshold,
        artifact_threshold_fallback,
        angle_threshold,
        eps,
        n_loops,
        ..Default::default()
    };

    neatnet_core::neatify(&mut network, &params, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Build output geometry column with input CRS metadata
    let ls_type = LineStringType::new(Dimension::XY, metadata);
    let builder = LineStringBuilder::from_line_strings(&network.geometries, ls_type);
    let ls_arr = builder.finish();
    let geom_ref: ArrayRef = ls_arr.to_array_ref();
    let geom_field = ls_arr.data_type().to_field("geometry", true);

    // Build status column
    let status_array: StringArray = network.statuses.iter().map(|s| Some(s.as_str())).collect();
    let status_ref: ArrayRef = Arc::new(status_array);
    let status_field = Field::new("status", DataType::Utf8, false);

    // Build parent_ids column as List<UInt64>
    use arrow::array::{ListBuilder, UInt64Builder};
    let mut parent_ids_builder = ListBuilder::new(UInt64Builder::new());
    for parents in &network.parent_ids {
        let values = parent_ids_builder.values();
        for &p in parents {
            values.append_value(p as u64);
        }
        parent_ids_builder.append(true);
    }
    let parent_ids_array = parent_ids_builder.finish();
    let parent_ids_ref: ArrayRef = Arc::new(parent_ids_array);
    let parent_ids_field = Field::new(
        "parent_ids",
        DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
        false,
    );

    let result_schema = Arc::new(Schema::new(vec![geom_field, status_field, parent_ids_field]));
    let result_batch = RecordBatch::try_new(result_schema.clone(), vec![geom_ref, status_ref, parent_ids_ref])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    PyTable::try_new(vec![result_batch], result_schema)?.into_pyarrow(py)
}

/// Run COINS continuity analysis on an Arrow table.
///
/// Returns the original table augmented with group, is_end,
/// stroke_length, and stroke_count columns.
#[pyfunction]
#[pyo3(signature = (table, angle_threshold=120.0))]
fn coins<'py>(
    py: Python<'py>,
    table: PyTable,
    angle_threshold: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let (geo_geoms, batch, schema, _metadata) = table_to_linestrings(table)?;
    let result = neatnet_core::continuity::coins(&geo_geoms, angle_threshold);

    // Build COINS columns
    let group_array: ArrayRef = Arc::new(Int64Array::from(
        result.group.into_iter().map(|v| v as i64).collect::<Vec<_>>(),
    ));
    let is_end_array: ArrayRef = Arc::new(BooleanArray::from(result.is_end));
    let stroke_length_array: ArrayRef = Arc::new(Float64Array::from(result.stroke_length));
    let stroke_count_array: ArrayRef = Arc::new(Int64Array::from(
        result
            .stroke_count
            .into_iter()
            .map(|v| v as i64)
            .collect::<Vec<_>>(),
    ));

    // Append COINS columns to existing table columns
    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();

    fields.push(Arc::new(Field::new("group", DataType::Int64, false)));
    fields.push(Arc::new(Field::new("is_end", DataType::Boolean, false)));
    fields.push(Arc::new(Field::new(
        "stroke_length",
        DataType::Float64,
        false,
    )));
    fields.push(Arc::new(Field::new(
        "stroke_count",
        DataType::Int64,
        false,
    )));

    columns.push(group_array);
    columns.push(is_end_array);
    columns.push(stroke_length_array);
    columns.push(stroke_count_array);

    let new_schema = Arc::new(Schema::new(fields));
    let result_batch = RecordBatch::try_new(new_schema.clone(), columns)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    PyTable::try_new(vec![result_batch], new_schema)?.into_pyarrow(py)
}

// ---------------------------------------------------------------------------
// WKT-based functions (for testing and backwards compatibility)
// ---------------------------------------------------------------------------

/// Simplify a street network (WKT interface).
#[pyfunction]
#[pyo3(signature = (
    wkt_geometries,
    *,
    edge_ids=None,
    max_segment_length=1.0,
    min_dangle_length=20.0,
    clip_limit=2.0,
    simplification_factor=2.0,
    consolidation_tolerance=10.0,
    artifact_threshold=None,
    artifact_threshold_fallback=7.0,
    angle_threshold=120.0,
    eps=1e-4,
    n_loops=2,
))]
fn neatify_wkt(
    py: Python<'_>,
    wkt_geometries: Vec<String>,
    edge_ids: Option<Vec<u64>>,
    max_segment_length: f64,
    min_dangle_length: f64,
    clip_limit: f64,
    simplification_factor: f64,
    consolidation_tolerance: f64,
    artifact_threshold: Option<f64>,
    artifact_threshold_fallback: f64,
    angle_threshold: f64,
    eps: f64,
    n_loops: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let geometries: Vec<LineString<f64>> = wkt_geometries
        .iter()
        .filter_map(|wkt| wkt_to_linestring(wkt))
        .collect();

    let n_input = geometries.len();
    let statuses = vec![neatnet_core::EdgeStatus::Original; n_input];
    let parent_ids: Vec<Vec<usize>> = match edge_ids {
        Some(ids) => {
            if ids.len() != n_input {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("edge_ids length ({}) != number of geometries ({})", ids.len(), n_input)
                ));
            }
            ids.into_iter().map(|id| vec![id as usize]).collect()
        }
        None => (0..n_input).map(|i| vec![i]).collect(),
    };

    let mut network = neatnet_core::StreetNetwork {
        geometries,
        statuses,
        parent_ids,
        attributes: None,
        crs: None,
    };

    let params = neatnet_core::NeatifyParams {
        max_segment_length,
        min_dangle_length,
        clip_limit,
        simplification_factor,
        consolidation_tolerance,
        artifact_threshold,
        artifact_threshold_fallback,
        angle_threshold,
        eps,
        n_loops,
        ..Default::default()
    };

    neatnet_core::neatify(&mut network, &params, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let geom_wkts: Vec<String> = network
        .geometries
        .iter()
        .map(|g| linestring_to_wkt(g))
        .collect();
    let status_strs: Vec<&str> = network.statuses.iter().map(|s| s.as_str()).collect();
    let parent_ids_out: Vec<Vec<u64>> = network
        .parent_ids
        .iter()
        .map(|ps| ps.iter().map(|&p| p as u64).collect())
        .collect();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("geometries", geom_wkts)?;
    dict.set_item("statuses", status_strs)?;
    dict.set_item("parent_ids", parent_ids_out)?;
    Ok(dict.into())
}

/// Run COINS continuity analysis (WKT interface).
#[pyfunction]
#[pyo3(signature = (wkt_geometries, angle_threshold=120.0))]
fn coins_wkt(
    py: Python<'_>,
    wkt_geometries: Vec<String>,
    angle_threshold: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let geometries: Vec<LineString<f64>> = wkt_geometries
        .iter()
        .filter_map(|wkt| wkt_to_linestring(wkt))
        .collect();

    let result = neatnet_core::continuity::coins(&geometries, angle_threshold);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("group", &result.group)?;
    dict.set_item("is_end", &result.is_end)?;
    dict.set_item("stroke_length", &result.stroke_length)?;
    dict.set_item("stroke_count", &result.stroke_count)?;
    dict.set_item("n_segments", result.n_segments)?;
    dict.set_item("n_p1_confirmed", result.n_p1_confirmed)?;
    dict.set_item("n_p2_confirmed", result.n_p2_confirmed)?;
    Ok(dict.into())
}

/// Compute voronoi skeleton from lines within a polygon (WKT interface).
#[pyfunction]
#[pyo3(signature = (wkt_lines, wkt_poly=None, wkt_snap_to=None, max_segment_length=1.0, clip_limit=2.0))]
fn voronoi_skeleton_wkt(
    py: Python<'_>,
    wkt_lines: Vec<String>,
    wkt_poly: Option<String>,
    wkt_snap_to: Option<Vec<String>>,
    max_segment_length: f64,
    clip_limit: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let lines: Vec<LineString<f64>> = wkt_lines
        .iter()
        .filter_map(|wkt| wkt_to_linestring(wkt))
        .collect();

    let poly = wkt_poly.and_then(|wkt| wkt_to_polygon(&wkt));
    let snap_geoms: Option<Vec<LineString<f64>>> = wkt_snap_to.map(|wkts| {
        wkts.iter()
            .filter_map(|wkt| wkt_to_linestring(wkt))
            .collect()
    });

    let (edgelines, splitters) = neatnet_core::geometry::voronoi_skeleton(
        &lines,
        poly.as_ref(),
        snap_geoms.as_deref(),
        max_segment_length,
        None,
        None,
        clip_limit,
        None,
    );

    let dict = pyo3::types::PyDict::new(py);
    let edge_wkts: Vec<String> = edgelines
        .iter()
        .map(|g| linestring_to_wkt(g))
        .collect();
    let split_wkts: Vec<String> = splitters
        .iter()
        .map(|g| linestring_to_wkt(g))
        .collect();
    dict.set_item("edgelines", edge_wkts)?;
    dict.set_item("splitters", split_wkts)?;
    Ok(dict.into())
}
