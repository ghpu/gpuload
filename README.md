GPULoad monitoring
==

GPULoad monitoring is a small executable for monitoring CUDA gpu utilization.
Depends on crate nv_ml.

Usage
==

ghpuload my_process

Note
==

GPULoad uses the process name to aggregate memory use. So if you launch a python script, please use :
 gpuload python my_script.py
instead of making your script executable, and launching 
 gpuload my_script.py 
directly.

