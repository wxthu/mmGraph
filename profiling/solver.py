
import gurobipy as gp
import json

def solver(model_num):
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