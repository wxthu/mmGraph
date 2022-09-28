
import os
import sqlite3
import json
import argparse

def post_process(index:int):
    cmd = ("ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed," +
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed " + 
            "--page raw --csv --import model.ncu-rep")
            
    result = os.popen(cmd)
    result = result.read().splitlines()

    compute_throughput = []
    memory_throughput = []
    kernel_latency = []

    for i in range(2, len(result)):
        element = result[i].split(',')
        compute_throughput.append(float(eval(element[-1])))
        memory_throughput.append(float(eval(element[-2])))


    connect = sqlite3.connect("model.sqlite")
    cursor = connect.cursor()

    for i, data in enumerate(cursor.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL")):
        duration = (data[1] - data[0]) * (1e-6)
        kernel_latency.append(duration)
        
    assert len(compute_throughput) == len(kernel_latency)

    properties = {"model_" + str(index) : {"compute":compute_throughput, "memory":memory_throughput, "latency":kernel_latency}}
    
    if os.path.isfile("properties.json"):
        with open("properties.json", "r") as f:
            data = json.load(f)
            data.update(properties)
            
        with open("properties.json", "w") as f:
            json.dump(data, f)
    else:
        with open('properties.json', 'w') as f:
            json.dump(properties, f)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help="Index of inference model")
    
    opt = vars(parser.parse_args())
    
    index = opt['index']
    post_process(index)