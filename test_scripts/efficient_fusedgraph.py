

import torch
import torchvision.models as models 
import nvtx

from profiling.solver import dp_solver

if __name__ == '__main__':
   
    model1 = models.alexnet().cuda()
    im1 = torch.randn(1, 3, 224, 224).cuda()
    model2 = models.squeezenet1_0().cuda()
    im2 = torch.randn(1, 3, 224, 224).cuda()
    
    groups = dp_solver(2)
    round = 200
    for _ in range(20):
        pred = model1(im1)
        pred = model2(im2)
        
    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(g1):
        out1 = model1(im1)
    
    with torch.cuda.graph(g2):
        out2 = model2(im2)
        
    fg = torch.cuda.CUDAFusedGraph([g1, g2], groups)
    fg.build_graph()
    with nvtx.annotate("fused_graph", color='red'):
        fg.launch_graph(round)
        torch.cuda.synchronize()
    
    
    
        
    
      
        
    
        
   
    
 