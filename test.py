import torch

x = torch.randn((59,300,300))

pred = torch.argmax(x, dim=0)
print(pred.shape)
out = torch.zeros_like(x).scatter_(0, pred.unsqueeze(0), 1.)
print(out.shape)