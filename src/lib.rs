use std::f32;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::ndarray::{Array2, Dim};
use rayon::prelude::*;

/**
 * Convert image space coords (r, c) to coordinate space (x, y) for a given dataset.
 */
#[inline]
fn to_image(gt: &Vec<f32>, x: f32, y: f32) -> [i32; 2] {
    [((y - gt[3]) / gt[5]) as i32, ((x - gt[0]) / gt[1]) as i32]
}

/**
 * Adjust the apparent height of an object at a certain distance, accounting for the curvature of the
 * earth and atmospheric refraction using the QGIS method.
 */
#[inline]
fn adjust_height(height: f32, distance_squared: f32, earth_diameter: f32, refraction_coefficient: f32) -> f32 {
    height - (distance_squared / earth_diameter) * (1.0 - refraction_coefficient)
}

/**
 * Run a single line of sight, returning the coordinates of cells that are visible.
 *
 * This function processes a single line of sight and returns a list of coordinates representing the visibility.
 */
fn process_line_of_sight(
    r0: i32, c0: i32, height0: f32, r1: i32, c1: i32, height1: f32, radius_px: f32,
    dem_data: &Array2<f32>, height: usize, width: usize
) -> Vec<(usize, usize)> {
    let mut visible_cells = Vec::new();
    let mut max_dydx = f32::NEG_INFINITY;

    for (r, c) in line_bresenham(r0, c0, r1, c1).skip(1) {
        let dx_squared = ((c0 - c).pow(2) + (r0 - r).pow(2)) as f32;
        let dx = dx_squared.sqrt();

        if dx > radius_px || r < 0 || r >= height as i32 || c < 0 || c >= width as i32 {
            break;
        }

        let base_dydx = (adjust_height(dem_data[(r as usize, c as usize)], dx_squared, 12740000.0, 0.13) - height0) / dx;
        let tip_dydx = (adjust_height(dem_data[(r as usize, c as usize)] + height1, dx_squared, 12740000.0, 0.13) - height0) / dx;

        if tip_dydx > max_dydx {
            visible_cells.push((r as usize, c as usize));
        }

        max_dydx = max_dydx.max(base_dydx);
    }

    visible_cells
}

/**
 * Calculate viewshed.
 */
#[pyfunction]
fn viewshed(py: Python, x0: f32, y0: f32, radius_m: f32, observer_height: f32, target_height: f32,
    dem_data: &PyArray2<f32>, gdal_transform: &PyTuple) -> PyResult<Py<PyArray2<i32>>> {
        
    let [height, width] = dem_data.shape() else { 
        panic!("DEM data not 2D")
    };
    
    let gt: Vec<f32> = gdal_transform.extract()?;
    let [r0, c0] = to_image(&gt, x0, y0);
    if r0 < 0 || r0 >= *height as i32 || c0 < 0 || c0 >= *width as i32 {
        panic!("Coordinates are out of bounds.");
    }

    let res = gt[1];
    let radius_px = (radius_m / res) as i32;
    let dem_data = dem_data.to_owned_array();
    let height0 = dem_data[(r0 as usize, c0 as usize)] + observer_height;

    let mut final_output = Array2::<i32>::zeros(Dim((*height, *width)));
    
    // Parallel processing of line of sight computations
    let visible_cells_lists: Vec<Vec<(usize, usize)>> = circle_perimeter(r0, c0, radius_px * 3)
        .par_iter()
        .map(|(r, c)| {
            process_line_of_sight(r0, c0, height0, *r, *c, target_height, radius_px as f32, &dem_data, *height, *width)
        })
        .collect();

    // Combine results from all threads by updating the output array with visible cells
    for visible_cells in visible_cells_lists {
        for (r, c) in visible_cells {
            final_output[(r, c)] = 1;
        }
    }

    let output_array = PyArray2::<i32>::from_owned_array(py, final_output);
    Ok(output_array.to_owned())
}

/**
 * Bresenham's line algorithm (as iterator).
 */
fn line_bresenham(x0: i32, y0: i32, x1: i32, y1: i32) -> impl Iterator<Item = (i32, i32)> {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut x = x0;
    let mut y = y0;

    std::iter::from_fn(move || {
        if x == x1 && y == y1 {
            return None;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
        return Some((y, x));
    })
}

/**
 * Bresenham's circle algorithm (octants).
 */
fn circle_perimeter(x0: i32, y0: i32, radius: i32) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        points.push((y0 + y, x0 + x));
        points.push((y0 + y, x0 - x));
        points.push((y0 - y, x0 + x));
        points.push((y0 - y, x0 - x));
        points.push((y0 + x, x0 + y));
        points.push((y0 + x, x0 - y));
        points.push((y0 - x, x0 + y));
        points.push((y0 - x, x0 - y));

        y += 1;
        err += 1 + 2 * y;
        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }

    points
}

/**
 * Export functions to Python.
 */
#[pymodule]
fn rust_viewshed(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(viewshed, m)?)?;
    Ok(())
}