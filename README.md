# py-viewshed-rust

This is a simple experiment in creating a Python library written in rust using a very basic viewshed algorithm (based on Bresenham's Line and Midpoint Circle algorithms, including atmospheric and curvature correction but no interpolation).

The interface to Python is handled using PyO3 and the wheel is built using `maturin`. 

This implementation is >32x faster than the same code implemented directly in Python.

Versions in which the lines of sight are calculated in parallel are included in the `parallel` branch (using synchronised access to the output object) and `parallel2` (which records the cells and then updates the output object later). Neither are much of an improvement on the original.

The commends required to install and run the are included in `build.sh` (I would run these yourself rather than using as a bash script).

To use the library, it is simply:

```python
import rasterio as rio
import rust_viewshed as vs
from matplotlib.pyplot import imshow, show

# load dataset
with rio.open('some-dem.tif') as ds:

    # get centre point
    l, b, r, t = ds.bounds
    centre_x = l + ((r - l) / 2)
    centre_y = b + ((t - b) / 2)

    # get viewshed
    result = vs.viewshed(x0=centre_x, y0=centre_y, radius_m=5000, observer_height=1.7, 
      target_height=10, dem_data=ds.read(1), gdal_transform=ds.transform.to_gdal())
    
    # display
    imshow(result)
    show()
```
