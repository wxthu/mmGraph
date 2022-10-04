

import torch
import torchvision.models as models 
import nvtx

from profiling.solver import dp_solver

if __name__ == '__main__':
   
    repos = [models.alexnet(), models.squeezenet1_0(), models.vgg11(), models.vgg13(),
              models.resnet.resnet18(), models.mobilenet_v2()]
    
    selected = [2, 3, 4]
    imgs = []
    graphs = []
    outs = [0] * len(selected)
    for i in range(len(selected)):
        repos[selected[i]].cuda()
        imgs.append(torch.randn(1, 3, 224, 224).cuda())
        graphs.append(torch.cuda.CUDAGraph())
    
    groups = dp_solver(selected)
    
    round = 1000
    for _ in range(20):
        for i in range(len(selected)):
            pred = repos[selected[i]](imgs[i])
  
    for j in range(len(selected)):
        with torch.cuda.graph(graphs[j]):
            outs[j] = repos[selected[j]](imgs[j])  
    
        
    fg3 = torch.cuda.CUDAFusedGraph(graphs)
    fg3.build_graph(2)
    range_id = nvtx.start_range("sequential", color='green')
    fg3.launch_graph(round)
    torch.cuda.synchronize()
    nvtx.end_range(range_id)  
    
    fg1 = torch.cuda.CUDAFusedGraph(graphs, groups)
    fg1.build_graph()
    range_id = nvtx.start_range("with_plan", color='red')
    fg1.launch_graph(round)
    torch.cuda.synchronize()
    nvtx.end_range(range_id)
        
    fg2 = torch.cuda.CUDAFusedGraph(graphs)
    fg2.build_graph(1)
    range_id = nvtx.start_range("no_plan", color='blue')
    fg2.launch_graph(round)
    torch.cuda.synchronize()
    nvtx.end_range(range_id)
    
     
    
    
    
        
    
      
        
    
        
   
    
 