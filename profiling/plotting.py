import os
import sqlite3
import numpy as np
import json
import matplotlib.pyplot as plt

import argparse

def sm_occ_plots(filename):
    connect = sqlite3.connect(filename)
    cursor = connect.cursor()
    
    metrics = ["Compute Warps In Flight", "Pixel Warps In Flight", "Unallocated Warps in Active SMs"]
    lTime = list()
    mData = [list() for m in metrics]
    
    for rawTimestamp, data in cursor.execute("SELECT rawTimestamp, data FROM GENERIC_EVENTS"):
        values = json.loads(data)
        for index, metric in enumerate(metrics):
            assert metric in values
            mData[index].append(int(values[metric]))
            
        lTime.append(rawTimestamp)

    connect.close()
    
    # filtering noise by Weining Chen
    ratio = 1000
    sinclen = 10
    sinc = np.sinc(np.linspace(-sinclen, sinclen, 2*sinclen*ratio+1))/ratio

    for metric, data in zip(metrics, mData):
        pdata = np.convolve(data, sinc)
        plt.plot(lTime[::ratio], pdata[sinclen*ratio:sinclen*ratio+len(lTime):ratio], label=metric)
        plt.legend(loc="upper left", fontsize = 5)
        
    plt.xlabel('Time/ns')
    plt.ylabel('SM occupancy')
    plt.savefig('SM_occupancy.jpg')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="model.sqlite", help="sqlite file derived from .rep file")
    
    opt = vars(parser.parse_args())
    
    filename = opt['file']
    sm_occ_plots(filename)