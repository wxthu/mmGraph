
import torch
import torchvision
import torchvision.models as models
import nvtx


def profile():
    warmup = models.alexnet().cuda()
    img = torch.randn(1, 3, 224, 224).cuda()
    
    repos = [warmup, models.squeezenet1_0().cuda(), models.vgg11().cuda(), models.vgg13().cuda(),
              models.resnet.resnet18().cuda(), models.mobilenet_v2().cuda()]
    
    for _ in range(50):
        pred = warmup(img)
    torch.cuda.synchronize()
    
    for i in range(len(repos)):
        range_id = nvtx.start_range("my_profiling"+str(i))
        output = repos[i](img)
        torch.cuda.synchronize()
        nvtx.end_range(range_id)
        
if __name__ == '__main__':
    profile()