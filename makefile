# Python library target
hgemm.cpython-312-x86_64-linux-gnu.so: setup.py
	uv run setup.py build_ext --inplace

# C++ benchmark target
bench: kernel/benchmark.cu
	nvcc -o bench kernel/benchmark.cu -lcublas -arch=sm_86 -std=c++17

# Build targets
build: hgemm.cpython-312-x86_64-linux-gnu.so

benchmark: bench

# Run benchmark and generate CSV
run-benchmark: bench
	./bench

# Plot results
plot: benchmark_results.csv
	uv run python plot.py

# Full workflow: build, run benchmark, and plot
full: bench hgemm.cpython-312-x86_64-linux-gnu.so
	./bench
	uv run python plot.py

all: build benchmark

# Clean targets
clean:
	rm -f hgemm.cpython-312-x86_64-linux-gnu.so bench benchmark_results.csv tflops_comparison.png

.PHONY: build benchmark run-benchmark plot full all clean