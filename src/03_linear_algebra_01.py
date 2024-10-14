# %matplotlib inline
import pandas as pd
import torch
from pandas.conftest import axis_1
from sympy.codegen import Print

## 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)

# x + y, x * y, x / y, x**y

print(x + y)
print(x * y)
print(x / y)
print(x**y)


# 向量
x = torch.arange(4)
print(x)

#indexing
print(x[3])

# 长度、维度和形状
len(x)
x.numel()

print(x.shape)



## 矩阵
# 正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶。
# 矩阵，我们通常用粗体、大写字母来表示（例如，X、Y和Z），在代码中表示为具有两个轴的张量。


A = torch.arange(20).reshape(5, 4)
print(A)

# 转置
print(A.T)

#symmetric matrix
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])

print(B == B.T)


# 张量
# 张量（tensor）是将向量推广到多个轴的数据结构。例如，向量是一阶张量，矩阵是二阶张量。

X = torch.arange(24).reshape(2, 3, 4)
print(X)


# 标量、向量、矩阵和任意数量轴的张量（本小节中的“张量”指代数对象）有一些实用的属性。例如，从按
# 元素操作的定义中可以注意到，任何按元素的一元运算都不会改变其操作数的形状。同样，给定具有相同形
# 状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。例如，将两个相同形状的矩阵相加，
# 会在这两个矩阵上执行元素加法。
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print(A)
print(A + B)

#具体而言，两个矩阵的按元素乘法称为Hadamard积（Hadamard product）（数学符号⊙）
print(A * B)

#将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a+X)
print((a*X).shape)


# 降维

# 我们可以对任意张量进行的一个有用的操作是计算其元素的和。数学表示法使用 符号表示求和。
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())

# 我们可以表示任意形状张量的元素和。例如，矩阵A中元素的和可以记为∑i∑jA[i,j]。
print(A.shape)
print(A.sum())

# 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。我们还可以指定张量沿哪一
# 个轴来通过求和降低维度。以矩阵为例，为了通过求和所有行的元素来降维（轴0），可以在调用函数时指
# 定axis=0。由于输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失。

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)


A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

# 沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。
# 结果和A.sum()相同
print(A.sum(axis=[0, 1]))

# 一个与求和相关的量是平均值（mean或average）。我们通过将总和除以元素总数来计算平均值。在代码中，
# 我们可以调用函数来计算任意形状张量的平均值。

print(A.mean())

print(A.sum() / A.numel())

# 同样，计算平均值的函数也可以沿指定轴降低张量的维度。
print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])


# 非降维求和
# 但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

# 例如，由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A。
print(A / sum_A)

# 如果我们想沿某个轴计算A元素的累积总和，比如axis=0（按行计算）
# ，可以调用cumsum函数。此函数不会沿
# 任何轴降低输入张量的维度。
print(A.cumsum(axis=0))

# 点积（Dot Product）
# 给定两个向量 x, y ∈ Rd，它们的点积x ⊤ y是相同位置的按元素乘积的和：x ⊤ y = ∑i=1d xi yi。

y = torch.ones(4, dtype = torch.float32)
print(x)
print(y)
print(torch.dot(x, y))

# 我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积：
print(torch.sum(x * y))

# 点积在很多场合都很有用。例如，给定一组由向量 x ∈ Rd 表示的值，和一组由 w ∈ Rd 表示的权重。
# x中的值根据权重w的加权和，可以表示为点积x ⊤ w。当权重为非负数且和为1（即  ∑wi=1 ）时，点积表示加权平均（weighted average)。
# 将两个向量规范化得到单位长度后，点积表示它们夹角的余弦。本节后面的内容将正式介绍长度（length）的概念。

# 矩阵-向量积
# A.shape, x.shape, torch.mv(A, x)
print(A.shape)
print(x.shape)
print(torch.mv(A, x))

# 矩阵-矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 范数

# 线性代数中最有用的一些运算符是范数（norm)。
# 非正式地说，向量的范数是表示一个向量有多大。这里考
# 虑的大小（size）概念不涉及维度，而是分量的大小。


## 向量范数
# 1.如果我们按常数因子α缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放
# 2.三角不等式：对于任意向量x和y，有f (x + y) ≤ f (x) + f (y).
# 3.范数必须是非负的: f (x) ≥ 0. (对所有x)，且当且仅当x=0时，f (x) = 0.


# L2范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数
print(torch.abs(u).sum())

# Frobenius 范数是一种矩阵范数，用来计算矩阵中所有元素的平方和的平方根。
# L2 范数通常指向量的欧几里得范数，用于计算向量的长度（或大小）。

# Frobenius 范数
print(torch.norm(torch.ones((4, 9))))

# 范数和目标
# 在深度学习中，我们经常试图解决优化问题：最大化分配给观测数据的概率; 最小化预测和真实观测之间的
# 距离。用向量表示物品（如单词、产品或新闻文章）
# ，以便最小化相似项目之间的距离，最大化不同项目之间
# 的距离。目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。


# len() 函数返回的是张量的第一个维度的长度，因此在这种情况下，len(X) 的输出为： len(X)=2
Z = torch.ones((2 ,3 ,4))
print(len(Z))

# print(A/A.sum(axis=1))

A = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状是 (2, 3)
B = torch.tensor([10, 20, 30])  # 形状是 (3,) 可以运算
C = torch.tensor([10, 20])  # 形状是 (1, 3) 可以运算
print(A + B)
print(A + C)