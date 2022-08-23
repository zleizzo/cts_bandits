import sys
import torch
from experiment_with_t import *

job = int(sys.argv[1])
# job = 0

# param_dict = {'hl_size':hl_sizes, 'reg':regs, 'lr':lrs, 'batch_size':batch_sizes, 'iters':iters}

hl_size = hl_sizes[job % len(hl_sizes)]
job = int(job / len(hl_sizes))

reg = regs[job % len(regs)]
job = int(job / len(regs))

lr = lrs[job % len(lrs)]
job = int(job / len(lrs))

batch_size = batch_sizes[job % len(batch_sizes)]
job = int(job / len(batch_sizes))

iters = iterss[job % len(iterss)]
job = int(job / len(iterss))

# # For testing
# hl_size = 30
# reg = 0.5
# lr = 0.1
# batch_size = 1
# iters = 10


net = BigNet(K, hl_size)
train(net, iters, reg, lr, batch_size, lam, p, K)

PATH = f'models/{hl_size},{reg},{lr},{batch_size},{iters}.pt'
torch.save(net.state_dict(), PATH)


# # Testing
# K = 4
# p = 2
# hl_size = 30
# reg = 1.
# lr = 0.1
# batch_size = 5
# lam = 0.1
# iters = 10

# net = BigNet(K, hl_size)
# train(net, iters, reg, lr, batch_size, lam, p, K)

# PATH = 'model.pt'
# torch.save(net.state_dict(), PATH)