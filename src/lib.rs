use std::f32;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::ndarray::{Array2, Dim}; // do not use ndarray directly!!

/**
 * Convert image space coords (r, c) to coordinate space (x, y) for a given dataset
 *    GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
 *    GT(1) w-e pixel resolution / pixel width.
 *    GT(2) row rotation (typically zero).
 *    GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
 *    GT(4) column rotation (typically zero).
 *    GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
 */
 #[inline]
 fn to_image(gt:&Vec<f32>, x:f32, y:f32) -> [i32; 2] {
    return [ ((y - gt[3]) / gt[5]) as i32, ((x - gt[0]) / gt[1]) as i32 ]
}

/**
 * Adjust the apparant height of an object at a certain distance, accounting for the curvature of the
 *  earth and atmospheric refraction using the QGIS method, see:
 *  https://www.zoran-cuckovic.from.hr/QGIS-visibility-analysis/help_qgis2.html#algorithm-options
 */
#[inline]
fn adjust_height(height: f32, distance_squared: f32, earth_diameter: f32, refraction_coefficient: f32) -> f32 {
    return height - (distance_squared / earth_diameter) * (1.0 - refraction_coefficient);
}

/**
 * Run a single line of sight, setting visible cells to 1 
 *
 * This function updates the putput surface directly by borrowing it (which might have implications 
 *  for future parallalisation)
 */
#[inline]
fn line_of_sight(r0: i32, c0: i32, height0: f32, r1: i32, c1: i32, height1: f32, radius_px: f32,
    dem_data: &Array2<f32>, height: usize, width: usize, output: &mut Array2<i32>) {

    // initialise max_dydx to -inf
    let mut max_dydx = f32::NEG_INFINITY;

    // loop through each cell in a line
    for (r, c) in line_bresenham(r0, c0, r1, c1).skip(1) {

        // calculate distance
         let dx_squared =((c0-c).pow(2) + (r0-r).pow(2)) as f32;
         let dx = dx_squared.sqrt();

        // stop if we run past the radius or off the end of the dataset
        if dx > radius_px || r < 0 || r >= height as i32 || c < 0 || c >= width as i32 {
            break;
        }

        // calculate the dy/dx ratios with and without the height added
        let base_dydx = (adjust_height(dem_data[(r as usize, c as usize)], dx_squared, 12740000.0, 0.13) - height0) / dx;
        let tip_dydx = (adjust_height(dem_data[(r as usize, c as usize)] + height1, dx_squared, 12740000.0, 0.13) - height0) / dx;

        // if tip dy/dx ratio is biggest so far, set cell to 1
        if tip_dydx > max_dydx {
            output[(r as usize, c as usize)] = 1;
        }

        // if base dy/dx is biggest so far, update it
        max_dydx = max_dydx.max(base_dydx);
    }
}

/**
 * calculate viewshed 
 */
#[pyfunction]
fn viewshed(py: Python, x0: f32, y0: f32, radius_m: f32, observer_height: f32, target_height: f32,
    dem_data: &PyArray2<f32>, gdal_transform: &PyTuple) -> PyResult<Py<PyArray2<i32>>> {
        
    // get dimensions of the dataset
    let [height, width] = dem_data.shape() else { 
        panic!("DEM data not 2D")
    };
    
    // extract transform data
    let gt: Vec<f32> = gdal_transform.extract()?;

    // convert coords to image space and verify that we are inside the dataset
    let [r0, c0] = to_image(&gt, x0, y0);
    if r0 < 0 || r0 >= *height as i32 || c0 < 0 || c0 >= *width as i32 {
        panic!("Coordinates are out of bounds.");
    }

    // convert radius to image space
    let radius_px = (radius_m / gt[1]) as i32;

    // extract dem_data to ndarray
    let dem_data = dem_data.to_owned_array();

    // get height of origin cell
    let height0 = dem_data[(r0 as usize, c0 as usize)] + observer_height;

    // init output surface and set origin to 1
    let mut output = Array2::<i32>::zeros(Dim((*height, *width)));
    output[(r0 as usize, c0 as usize)] = 1;

    // run a line of sight to each point in the perimeter
    for (r, c) in circle_perimeter(r0, c0, radius_px * 3) { // TODO: is *3 necessary / best?
        line_of_sight(r0, c0, height0, r, c, target_height, radius_px as f32, &dem_data, *height, *width, &mut output);
    }

    // Convert to PyArray2 and return
    let output_array = PyArray2::<i32>::from_owned_array(py, output);
    return Ok(output_array.to_owned());
}

/**
 * Bresenham's line algorithm (as iterator)
 */
#[inline]
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
 * Bresenham's circle algorithm (octants)
 */
#[inline]
fn circle_perimeter(x0: i32, y0: i32, radius: i32) -> Vec<(i32, i32)> {
    
    // initialise new vector
    let mut points = Vec::new();
    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    // calculate circle perimeter in octents
    while x >= y {
        points.push((y0 + y, x0 + x));
        points.push((y0 + y, x0 - x));
        points.push((y0 - y, x0 + x));
        points.push((y0 - y, x0 - x));
        points.push((y0 + x, x0 + y));
        points.push((y0 + x, x0 - y));
        points.push((y0 - x, x0 + y));
        points.push((y0 - x, x0 - y));

        // update location and error
        y += 1;
        err += 1 + 2 * y;
        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }

    // return points vector
    return points;
}

/**
 * export functions to Python
 */
#[pymodule]
fn rust_viewshed(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(viewshed, m)?)?;
    return Ok(());
}
