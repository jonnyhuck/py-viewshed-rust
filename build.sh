# make sure that you have this
pip install maturin

# set up project
cargo new --lib rust_viewshed
cd rust_viewshed

# build rust lib - DON'T NEED THIS??
cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup

# build wheel
maturin develop