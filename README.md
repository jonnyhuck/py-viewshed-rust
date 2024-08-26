# py-viewshed-rust

This is a simple experiment in creating a Python library written in rust using a very basic viewshed algorithm (based on Bresenham's Line and Midpoint Circle algorithms, including atmospheric and curvature correction but no interpolation).

The interface to Python is handled using PyO3 and the wheel is built using `maturin`. 

This implementation is >32x faster than the same code implemented directly in Python.

Versions in which the lines of sight are calculated in parallel are included in the `parallel` branch (using synchronised access to the output object) and `parallel2` (which records the cells and then updates the output object later). Neither are much of an improvement on the original.

The commends required to install and run the are included in `build.sh` (I would run these yourself rather than using as a bash script).
