
import gurobipy as gp
import itertools
import numpy as np
import json

def ilp_solver(model_num):
    # The number of models
    N = model_num

    # compute throught, memory throughput and latency for each kernel in different models
    c = list()
    m = list()
    l = list()

    nKernel = list()

    with open("../profiling/properties.json", "r") as f:
        data = json.load(f)
        for key, val in data.items():
            c.append(val['compute'])
            m.append(val['memory'])
            l.append(val['latency'])
            nKernel.append(len(val['latency']))

    # remove noise in profiling data
    for i in range(N):
        for j in range(len(c[i])):
            if c[i][j] >= 100.0:
                c[i][j] = 99.9
            if m[i][j] >= 100.0:
                m[i][j] = 99.9
                
    K = sum(nKernel)
    MODEL = gp.Model()
    
    x = list()
    for i in range(N):
        x.append(MODEL.addVars(nKernel[i], K, vtype=gp.GRB.BINARY, name="x_%d"%(i)))

    latency_g = MODEL.addVars(K, vtype=gp.GRB.CONTINUOUS, name="group_latency")
    total_latency = MODEL.addVar(vtype=gp.GRB.CONTINUOUS, name="total_latency")

    MODEL.update()

    MODEL.setObjective(total_latency, sense=gp.GRB.MINIMIZE)

    MODEL.addConstr(total_latency == gp.quicksum(latency_g[i] for i in range(K)))

    for k in range(K):
        for i in range(N):
            MODEL.addConstr(gp.quicksum(l[i][j] * x[i][j, k] for j in range(nKernel[i])) <= latency_g[k])

    for i in range(N):
        for j in range(nKernel[i]):
            MODEL.addConstr(gp.quicksum(x[i][j, k] for k in range(K)) == 1)
            
    for i in range(N):
        for j in range(1, nKernel[i]):
            for k in range(1, K):
                MODEL.addConstr(gp.quicksum(x[i][j-1, delta] for delta in range(k)) >= x[i][j, k])
                
    for k in range(K):
        MODEL.addConstr(gp.quicksum(c[i][j] * x[i][j, k] for i in range(N) for j in range(nKernel[i])) <= 100.0)
        MODEL.addConstr(gp.quicksum(m[i][j] * x[i][j, k] for i in range(N) for j in range(nKernel[i])) <= 100.0)
        
    MODEL.optimize()
    
def dp_solver(model_list):
    N = len(model_list)
    # compute throught, memory throughput and latency for each kernel in different models
    c = list()
    m = list()
    l = list()

    nKernel = list()

    with open("../profiling/properties.json", "r") as f:
        data = json.load(f)
        for i, (key, val) in enumerate(data.items()):
            if not i in model_list:
                continue
            c.append([0.0] + val['compute'])
            m.append([0.0] + val['memory'])
            l.append([0.0] + val['latency'])
            nKernel.append(len(val['latency']))

    # remove noise in profiling data
    for i in range(N):
        for j in range(len(c[i])):
            if c[i][j] >= 100.0:
                c[i][j] = 99.9
            if m[i][j] >= 100.0:
                m[i][j] = 99.9
    
    dp = np.zeros(list(x+1 for x in nKernel))    
    for i in range(len(nKernel)):
        lIndex = [0] * len(nKernel)
        lIndex[i] = slice(None, None, None)
        lat = [sum(l[i][:k+1]) for k in range(0, nKernel[i]+1)]
        dp[tuple(lIndex)] = np.array(lat)       
    
    def calc_tmp(iCurr, iSolo):
        iPrev = [x-y if x-y > 0 else 0 for x, y in zip(iCurr, iSolo)]
        pairs = [[i, iCurr[i]]for i in range(N) if iSolo[i] != 0 and iCurr[i] != 0]
        
        if len(pairs) > 0 \
            and sum([c[i][j] for i, j in pairs]) <= 100.0 \
            and sum([m[i][j] for i, j in pairs]) <= 100.0:
            return dp[tuple(iPrev)] + max([l[i][j] for i, j in pairs])
        else:
            return float("inf")   
            
    for iCurr in itertools.product(*[range(0, x+1) for x in nKernel]):
        if iCurr.count(0) >= N-1:
            continue
        
        states = []
        for iSolo in itertools.product([0, 1], repeat=N):
            if any(iSolo):
                tmp = calc_tmp(iCurr, iSolo)
                states.append(tmp)
                
        dp[tuple(iCurr)] = min(states)
            
    # reversed lookup
    groups = []
    next_id = None
    for iCurr in itertools.product(*[range(x, -1, -1) for x in nKernel]):
        if sum(iCurr) == 0:
            break
        
        if next_id is not None:
            if iCurr != next_id:
                continue
                
        for iSolo in itertools.product([0, 1], repeat=N):
            if any(iSolo):
                if dp[tuple(iCurr)] == calc_tmp(iCurr, iSolo):
                    groups.append(iSolo)
                    next_id = tuple([x-y if x-y > 0 else 0 for x, y in zip(iCurr, iSolo)])
                    break
                
    groups.reverse()
    return groups
    