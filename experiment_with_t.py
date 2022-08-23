import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Net(nn.Module):
    def __init__(self, K, hl_size):
        super().__init__()
        self.fc1 = nn.Linear(2 * K + 1, hl_size)
        self.fc3 = nn.Linear(hl_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc3(x)
        return x


class BigNet(nn.Module):
    def __init__(self, K, hl_size):
        super().__init__()
        self.fc1 = nn.Linear(2 * K + 1, hl_size)
        self.fc2 = nn.Linear(hl_size, hl_size)
        self.fc3 = nn.Linear(hl_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def sample(K, t=None):
    # K = problem constant
    # Returns states in the form [s_1, ..., s_K, q_1, ..., q_{K-1}, t]
    if t is None:
        t = torch.rand(1)
    q = torch.rand(K)
    s = 2 * torch.rand(K) - 1.
    
    q *= t / torch.sum(q)
    s *= q
    return t, s, q


def compute_mu_hat(s, q):
    # Returns a length K vector of mu hat values
    return (2. + s) / (2. + q)


def compute_grad_V(net, state):
    V = net(state)
    grad_V = torch.autograd.grad(V, state, create_graph = True)[0]
    return grad_V


def compute_logsumexp_terms(net, state, K):
    grad_V = compute_grad_V(net, state)
    ds2 = torch.zeros(K)
    for k in range(K):
        ds2[k] = torch.autograd.grad(grad_V[k], state, create_graph = True)[0][k]

    s = state[:K]
    q = state[K:2*K]
    mu_hat = compute_mu_hat(s, q)

    # mu_k * ds_k V + dq_k V + (1/2) ds_k^2 V + mu_k
    logsumexp_terms = (mu_hat * grad_V[:K] + grad_V[K:2*K] + ds2 / 2. + mu_hat)

    return logsumexp_terms


def hjb_term(net, t, s, q, lam, p, K):
    state = torch.hstack([s, q, t])
    state.requires_grad = True

    grad_V = compute_grad_V(net, state)
    logsumexp_terms = compute_logsumexp_terms(net, state, K)

    return torch.pow(torch.abs(grad_V[0] + lam * torch.logsumexp(logsumexp_terms / lam, 0)), p)


def hjb_bdry(net, s, q, p):
    state = torch.hstack([s, q, torch.tensor(1.)])
    state.requires_grad = True

    V = net(state)
    return torch.pow(torch.abs(V), p)
    


def train(net, iters, reg, lr, batch_size, lam, p, K):
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for i in range(iters):
        loss = 0.
        bdry_loss = 0.
        for j in range(batch_size):
            t, s, q = sample(K)
            _, s_bdry, q_bdry = sample(K, t=1.)

            loss += hjb_term(net, t, s, q, lam, p, K)
            bdry_loss += hjb_bdry(net, s_bdry, q_bdry, p)
        
        total_loss = (loss + reg * bdry_loss) / batch_size

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return net


def policy(net, t, s, q, K):
    state = torch.hstack([s, q, torch.tensor(1.)])
    state.requires_grad = True
    softmax_inputs = compute_logsumexp_terms(net, state, K)
    return F.softmax(softmax_inputs, 0)



# Constants
K = 4
p = 2
lam = 0.1


hl_sizes = [30, 60, 120]
regs = [1., 0.3, 0.1, 0.03]
lrs = [1., 0.3, 0.1, 0.03, 0.01]
batch_sizes = [1, 5, 25]
iterss = [100000, 1000000, 10000000, 100000000]