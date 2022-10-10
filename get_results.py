import torch
import os
import matplotlib as plt
import numpy as np

# from experiment_with_t import BigNet, sample, K, p, lam, hl_sizes, regs, lrs, batch_sizes, iterss
from experiment_with_t import *


# hl_sizes = [30, 60, 120]
# regs = [1., 0.3, 0.1, 0.03]
# lrs = [1., 0.3, 0.1, 0.03, 0.01]
# batch_sizes = [1, 5, 25]
# iterss = [100000, 1000000, 10000000, 100000000]



n = 1000
samples = []
bdry_samples = []
for i in range(n):
    samples.append(sample(K))
    bdry_samples.append(sample(K, t=1.))

min_loss = float('inf')
for hl_size in hl_sizes:
    for reg in regs:
        for lr in lrs:
            for batch_size in batch_sizes:
                for iters in iterss:
                    PATH = f'models/{hl_size},{reg},{lr},{batch_size},{iters}.pt'
                    net = BigNet(K, hl_size)
                    net.load_state_dict(torch.load(PATH))
                    
                    loss = 0
                    bdry_loss = 0
                    for sample, bdry_sample in zip(samples, bdry_samples):
                        t, s, q = sample
                        bdry_s, bdry_q = bdry_sample
                        loss += hjb_term(net, t, s, q, lam, p)
                        bdry_loss += hjb_bdry(net, bdry_s, bdry_q, p)
                    
                    total_loss = loss + bdry_loss
                    if total_loss < min_loss:
                        min_loss = total_loss
                        best_net = net.clone()
                        best_params = PATH


print(min_loss)
print(best_params)