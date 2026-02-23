//! Heap-profiling binary using dhat.
//! Usage: cargo run --release --features dhat-heap --bin profile_memory -- /path/to/edges.[wkt|parquet] [n_loops]
//!
//! Produces dhat-heap.json on exit; view at https://nnethercote.github.io/dh_view/dh_view.html

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::fs;
use std::path::Path;
use std::time::Instant;

use geo_types::LineString;
use neatnet_core::types::{EdgeStatus, NeatifyParams, StreetNetwork};

fn load_wkt(path: &str) -> Vec<LineString<f64>> {
    let content = fs::read_to_string(path).expect("Failed to read WKT file");
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            use std::str::FromStr;
            wkt::Wkt::from_str(l)
                .ok()
                .and_then(|w| geo_types::Geometry::try_from(w).ok())
                .and_then(|g| match g {
                    geo_types::Geometry::LineString(ls) => Some(ls),
                    _ => None,
                })
        })
        .collect()
}

fn load_parquet(path: &str) -> Vec<LineString<f64>> {
    use arrow::array::{Array, AsArray};

    let file = fs::File::open(path).expect("Failed to open Parquet file");
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to build Parquet reader");
    let reader = builder.build().expect("Failed to build batch reader");

    let mut geometries: Vec<LineString<f64>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.expect("Failed to read record batch");
        let schema = batch.schema();

        let geom_col_idx = schema
            .column_with_name("geometry")
            .map(|(i, _)| i)
            .unwrap_or(0);

        let array_ref = batch.column(geom_col_idx);
        let binary_array = array_ref.as_binary::<i32>();
        for i in 0..binary_array.len() {
            if binary_array.is_null(i) {
                continue;
            }
            let wkb = binary_array.value(i);
            if let Some(ls) = parse_wkb_linestring(wkb) {
                geometries.push(ls);
            }
        }
    }

    geometries
}

fn parse_wkb_linestring(wkb: &[u8]) -> Option<LineString<f64>> {
    use geo_types::Coord;
    if wkb.len() < 9 {
        return None;
    }
    let le = wkb[0] == 1;
    let geom_type = if le {
        u32::from_le_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    } else {
        u32::from_be_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    };
    if geom_type != 2 {
        return None;
    }
    let n_points = if le {
        u32::from_le_bytes([wkb[5], wkb[6], wkb[7], wkb[8]]) as usize
    } else {
        u32::from_be_bytes([wkb[5], wkb[6], wkb[7], wkb[8]]) as usize
    };
    let expected_len = 9 + n_points * 16;
    if wkb.len() < expected_len {
        return None;
    }
    let mut coords = Vec::with_capacity(n_points);
    let mut offset = 9;
    for _ in 0..n_points {
        let x = if le {
            f64::from_le_bytes(wkb[offset..offset + 8].try_into().ok()?)
        } else {
            f64::from_be_bytes(wkb[offset..offset + 8].try_into().ok()?)
        };
        offset += 8;
        let y = if le {
            f64::from_le_bytes(wkb[offset..offset + 8].try_into().ok()?)
        } else {
            f64::from_be_bytes(wkb[offset..offset + 8].try_into().ok()?)
        };
        offset += 8;
        coords.push(Coord { x, y });
    }
    Some(LineString::new(coords))
}

fn main() {
    let _profiler = dhat::Profiler::new_heap();

    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .expect("Usage: profile_memory <edges.wkt|edges.parquet> [n_loops]");
    let n_loops = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    eprintln!("Loading from {}", input_path);
    let t0 = Instant::now();

    let geometries = if Path::new(input_path)
        .extension()
        .map_or(false, |ext| ext == "parquet")
    {
        load_parquet(input_path)
    } else {
        load_wkt(input_path)
    };

    eprintln!(
        "Loaded {} edges in {:.3}s",
        geometries.len(),
        t0.elapsed().as_secs_f64()
    );

    let n_edges = geometries.len();
    let statuses = vec![EdgeStatus::Original; n_edges];
    let parent_ids: Vec<Vec<usize>> = (0..n_edges).map(|i| vec![i]).collect();
    let mut network = StreetNetwork {
        geometries,
        statuses,
        parent_ids,
        attributes: None,
        crs: None,
    };

    let params = NeatifyParams {
        n_loops,
        ..NeatifyParams::default()
    };

    eprintln!("Running neatify with n_loops={}...", n_loops);
    let t0 = Instant::now();
    neatnet_core::simplify::neatify(&mut network, &params, None).unwrap();
    eprintln!("neatify completed in {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("Output: {} edges", network.geometries.len());
}
