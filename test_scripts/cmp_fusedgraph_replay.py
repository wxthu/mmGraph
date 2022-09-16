

from operator import mod
import torch
import torch.nn as nn
import nvtx

class Customized(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU()
        )
        
        for _ in range(100):
            self.layers.append(nn.Conv2d(1, 1, 1))
            self.layers.append(nn.ReLU())
    
    def forward(self, x):
        out = self.layers(x)
        return out

if __name__ == '__main__':
   
    model1 = Customized().cuda()
    im1 = torch.randn(1, 1, 50, 50).cuda()
    model2 = Customized().cuda()
    im2 = torch.randn(1, 1, 50, 50).cuda()
    
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
        
    fg = torch.cuda.CUDAFusedGraph()
    fg.build_graph([g1, g2])
    with nvtx.annotate("fused_graph", color='red'):
        fg.launch_graph(round)
        torch.cuda.synchronize()
    
    with nvtx.annotate("single_replay", color='blue'):
        for _ in range(round):
            g1.replay()
        torch.cuda.synchronize()
        
    
      
        
    
        
   
    
 