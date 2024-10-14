# %matplotlib inline
import pandas as pd
import torch
import numpy as np
from pandas.conftest import axis_1
from sympy.codegen import Print

from matplotlib_inline import backend_inline
from d2l import torch as d2l

# 事实上，逼近法就是积分（integral calculus）的起源。
# 2000多年后，微积分的另一支，微分（differential calculus）被发明出来。
# 微分和积分是微积分的两个主要分支。



# 拟合模型的任务分解为两个关键问题：
# 优化（optimization）：用模型拟合观测数据的过程；
# 泛化（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    # print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

# 为了对导数的这种解释进行可视化，我们将使用matplotlib，这是一个Python中流行的绘图库。
# 要配置matplotlib生成图形的属性，我们需要定义几个函数。
# 在下面，use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像。
# 注意，注释#@save是一个特殊的标记，会将对应的函数、类或语句保存在d2l包中。
# 因此，以后无须重新定义就可以直接调用它们（例如，d2l.use_svg_display()）。

def use_svg_display(): #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

# 我们定义set_figsize函数来设置图表大小。
# 注意，这里可以直接使用d2l.plt，
# 因为导入语句 from matplotlib import pyplot as plt已标记为保存到d2l包中。

def set_figsize(figsize=(3.5, 2.5)): #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

# 下面的set_axes函数用于设置由matplotlib生成图表的轴的属性。

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

# 通过这三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线，因为我们需要在整个书中可视化许多曲线。
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
    ylim=None, xscale='linear', yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
    else:
        axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)



# 现在我们可以绘制函数f(x)和它在x = 1处的切线y = 2x - 3。
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

# 偏导数
# 我们只讨论了仅含一个变量的函数的微分。在深度学习中，函数通常依赖于许多变量。因此，我
# 们需要将微分的思想推广到多元函数（multivariate function）上。这就涉及到偏导数（partial derivative）的概念。