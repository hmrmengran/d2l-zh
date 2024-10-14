# %matplotlib inline
import random
import torch
import numpy
from sympy.printing.codeprinter import requires

from d2l import torch as d2l
import matplotlib.pyplot as plt



def synthetic_data(w, b, num_examples):
    " 生成 y = xw + b + noise "
    X = torch.normal(0, 1, (num_examples, len(w)))
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.01, Y.shape)
    return X ,Y.reshape((-1, 1))
true_w  = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\n label:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor((indices[i:min(i + batch_size, num_examples)]))
        yield features[batch_indices], labels[batch_indices]

batch_size =10
for X, Y in data_iter(batch_size, features, labels):
    print(X, '\n', Y)
    break


w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 模型定义
def linreg(x, w, b):
    '线性回归模型'
    return torch.matmul(x, w)+b

# lost func
def squared_loss(y_hat, y):
    '均方损失'
    return (y_hat - y.reshape(y_hat.shape))**2 /2

# 定义优化算法

def sgd(params, lr, batch_size):
    '小批量随机剃度下降'
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X , y in data_iter(batch_size, features, labels):
        # X and y 小批量loss
        l = loss(net(X, w, b), y)
        # 因为 l的形状是[batch_size, l] ，而不是一个标量。 ‘l’中的所有元素被加到一起
        #并以此计算关于[w, b] 的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_1 = loss(net(features, w, b), labels)
        print(f'epoch {epoch +1}, loss {float(train_1.mean()):f}')


# 测试不同的 lr 以及mum_epochs, batch_size

