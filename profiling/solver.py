
import gurobipy as gp
import json

def ilp_solver(model_num):
    # The number of models
    N = model_num

    # compute throught, memory throughput and latency for each kernel in different models
    c = list()
    m = list()
    l = list()

    nKernel = list()

    with open("properties.json", "r") as f:
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
    
def dp_solver(model_num):
    N = model_num
    # compute throught, memory throughput and latency for each kernel in different models
    c = list()
    m = list()
    l = list()

    nKernel = list()

    with open("../profiling/properties.json", "r") as f:
        data = json.load(f)
        for i, (key, val) in enumerate(data.items()):
            if i < 4:
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
                
    dp = [[0.0 for j in range(nKernel[1] + 1)] for i in range(nKernel[0] + 1)]
    for j in range(1, nKernel[1] + 1):
        dp[0][j] = sum([l[1][k] for k in range(1, j+1)])
    
    for j in range(1, nKernel[0] + 1):
        dp[j][0] = sum([l[0][k] for k in range(1, j+1)])
        
    def calc_tmp(i, j):
        lat1 = dp[i-1][j] + l[0][i]
        lat2 = dp[i][j-1] + l[1][j]
        
        if c[0][i] + c[1][j] <= 100.0 and m[0][i] + m[1][j] <= 100.0:
            lat3 = dp[i-1][j-1] + max(l[0][i], l[1][j])
        else:
            lat3 = float("inf")
            
        return lat1, lat2, lat3
            
    for i in range(1, nKernel[0] + 1):
        for j in range(1, nKernel[1] + 1):
            lat1, lat2, lat3 = calc_tmp(i, j)
            dp[i][j] = min(lat1, lat2, lat3)
            
    # reversed lookup
    g1 = [0 for _ in range(nKernel[0] + 1)]
    g2 = [0 for _ in range(nKernel[1] + 1)]
    row = nKernel[0]
    col = nKernel[1]
    gIndex = 1
    
    while row != 0 and col != 0:
        tmp1, tmp2, tmp3 = calc_tmp(row, col)
        lat = dp[row][col]
        if lat == tmp1:
            g1[row] = gIndex
            row -= 1
        elif lat == tmp2:
            g2[col] = gIndex
            col -= 1
        elif lat == tmp3:
            g1[row] = gIndex
            g2[col] = gIndex 
            row -= 1
            col -= 1 
        else: 
            assert False
        gIndex += 1
    
    while row > 0:
        g1[row] = gIndex
        gIndex += 1
        row -= 1
    while col > 0:
        g2[col] = gIndex
        gIndex += 1
        col -= 1
     
    # merge
    cursor1 = 1
    cursor2 = 1

    groups = []
    while cursor1 < len(g1) and cursor2 < len(g2):
        if g1[cursor1] == g2[cursor2]:
            groups.append([0, 1])
            cursor1 += 1
            cursor2 += 1
        elif g1[cursor1] > g2[cursor2]:
            groups.append([0])
            cursor1 += 1
        else:
            groups.append([1])
            cursor2 += 1
            
    while cursor1 < len(g1):
        groups.append([0])
        cursor1 += 1
        
    while cursor2 < len(g2):
        groups.append([1])
        cursor2 += 1
        
    return groups
    