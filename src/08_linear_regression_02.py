# %matplotlib inline
import torch
import numpy as np
from sympy.integrals.manualintegrate import rewrites_rule
from torch.utils import data
from torch import nn
from sympy.printing.codeprinter import requires

from d2l import torch as d2l
import matplotlib.pyplot as plt

# 线性回归的简洁实现 使用pytorch nn module

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    '构造一个Pytorch数据迭代器'
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels),batch_size)
next(iter(data_iter))



# 'nn'是神经为网络的缩写
# 也可以直接使用nn.Linear nn.Sequential是一个容器
net = nn.Sequential(nn.Linear(2, 1))


#初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

print("net:", net[0])

# use MSELoss
loss = nn.MSELoss()

#实例化SGD
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 3
true_w = torch.tensor([2, -3.4])
true_b = 4.2
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l =loss(net(features), labels)
    print(f'epoch {epoch +1}, loss {l:f}')



