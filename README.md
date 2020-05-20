GPULoad monitoring
==

GPULoad monitoring is a linux only small executable for monitoring CUDA gpu utilization.

It needs to be compiled on a machine with libnvidia-ml installed.


Build instructions
==

    cargo build --release
    strip target/release/gpuload


Usage
==

ghpuload my_process

