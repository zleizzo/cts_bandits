import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def mu_hat(x):
    return (1. + x[1]) / (1. + x[2])

# dtV + max_pi [ \sum_k mu_hat(s_k, q_k) ds_kV + dq_kV + sigma_hat^2/2 + mu_hat(s_k, q_k) ]
def hjb_term(net, x):
    outputs = net(x)
    grad_v = torch.autograd.grad(outputs, x, create_graph = True)[0]
    return grad_v[0] + torch.max(mu_hat(x) * (grad_v[1] + 1.) + grad_v[2] + 0.5, torch.tensor([0.5]))


def train(net, n, reg, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for i in range(n):
        t = torch.rand(1)
        q = t * torch.rand(1)
        s = 3 * torch.rand(1)
        x = torch.tensor([t, s, q], requires_grad = True)
        bdry_x = torch.tensor([1., s, q], requires_grad = True)

        optimizer.zero_grad()
        loss = torch.pow(hjb_term(net, x), 2) + reg * torch.pow(net(bdry_x), 2)
        loss.backward()
        optimizer.step()


# n = 10000
# lr = 0.01
# reg = 0.01
# size = 'small'

ns = [1000, 10000, 100000]
lrs = [0.001, 0.01, 0.1, 0.5]
regs = [0.001, 0.01, 0.1, 1.]
sizes = ['small', 'big']

# job = int(sys.argv[1])
job = 0

n = ns[job % len(ns)]
job = int(job / len(ns))

lr = lrs[job % len(lrs)]
job = int(job / len(lrs))

reg = regs[job % len(regs)]
job = int(job / len(regs))

size = sizes[job % len(sizes)]


if size == 'small':
    net = Net()
else:
    assert size == 'big'
    net = BigNet()


train(net, n, reg, lr)

# PATH = f'models/{n},{lr},{reg},{size}.pt'
PATH = 'model.pt'
torch.save(net.state_dict(), PATH)


# net = Net()
# net.load_state_dict(torch.load(PATH))