#!/bin/bash

export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

for((i=0;i<6;++i));
do 
nsys profile -f true --capture-range=nvtx --nvtx-capture=my_profiling${i} -o model python profiling.py
nsys export -f true -t sqlite model.nsys-rep
sudo $(which ncu) -o model -f --nvtx --nvtx-include=my_profiling${i} $(which python) profiling.py
python extract_data.py --index=${i}
done
