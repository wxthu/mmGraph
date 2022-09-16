
'''
This script is mainly responsible for comparing the performance
difference between normal forward process and graph reply
'''

import torch
import torch.nn as nn
import nvtx

class Customized(nn.Module):
    def __init__(self):
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
   
    model = Customized().cuda()
    im = torch.randn(1, 1, 50, 50).cuda()
    
    round = 500
    for _ in range(50):
        pred = model(im)
        
    with nvtx.annotate("forward", color='red'):
        for _ in range(round):
            pred = model(im)
        torch.cuda.synchronize()
        
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(im)
    
    with nvtx.annotate("replay", color='blue'):
        for _ in range(round):
            g.replay()
        torch.cuda.synchronize()
        
    
      
        
    
        
   
    
 