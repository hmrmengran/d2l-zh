import torch
from IPython import display
from d2l import torch as d2l

# 随机读取256张图片
batch_size = 256

# 返回训练集和测试集的迭代器
# load_data_fashion_mnist函数是在图像分类数据集中定义的一个函数，可以返回batch_size大小的训练数据集和测试数据集
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#展平长度为784的向量
num_inputs = 784
#10输出，也对应10类别
num_outputs = 10

# 权重w：均值为0，标准差为0.01，数量size为输入输出的数量
# size=(num_inputs, num_outputs)：行数为输入的个数，列数等于输出的个数
# requires_grad=True表明要计算梯度
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
#分别以维度0和1求和,6=【1，2，3】相加 15=【4，5，6】自己相加
X.sum(0,keepdim=True),X.sum(1,keepdim=True)

# 定义softmax函数
def softmax(X):
    # 对每个元素做指数运算
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    #广播机制，每一项除以总和，求概率，每行概率和为1
    return X_exp / partition

X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
X_prob,X_prob.sum(1)

def net(X):
    # 权重为784×10的矩阵，这里将原始图像的列数大小转换为权重w矩阵的行数大小
    # 模型简单看来为：softmax(wx' + b)
  return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
#[0,1]对应y[0]->0.y[1]->2;
#从中选取y_hat[0,0]->0.1,y_hat[1,2]->0.5
y_hat[[0,1],y]


def cross_entropy(y_hat,y):
  #y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])得出len(y_hat)=2
  #-torch.log(y_hat[0][0])、-torch.log(y_hat[1][2])
  return -torch.log(y_hat[range(len(y_hat))


def accuracy(y_hat, y):
  """计算预测正确的数量"""
  "len是查看矩阵的行数"
  "y_hat.shape[1]就是去列数"
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
      "第2个维度为预测标签，取最大元素"
      y_hat = y_hat.argmax(axis=1)

  "将y_hat转换为y的数据类型然后作比较，cmp函数存储bool类型"
  cmp = y_hat.type(y.dtype) == y
  return float(cmp.type(y.dtype).sum())  # 将正确预测的数量相加


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    # 判断模型是否为深度学习模型
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # metric：度量，累加正确预测数、预测总数

    # 梯度不需要反向传播
    with torch.no_grad():
        # 每次从迭代器中拿出一个X和y
        for X, y in data_iter:
            # metric[0, 1]分别为网络预测正确的数量和总预测的数量
            # nex(X)：X放在net模型中进行softmax操作
            # numel()函数：返回数组中元素的个数，在此可以求得样本数
            metric.add(accuracy(net(X), y), y.numel())

        # # metric[0, 1]分别为网络预测正确数量和总预测数量
    return metric[0] / metric[1]


class Accumulator:  # @save
    """在n个变量上累加"""

    # 初始化根据传进来n的大小来创建n个空间，全部初始化为0.0
    def __init__(self, n):
        self.data = [0.0] * n

    # 把原来类中对应位置的data和新传入的args做a + float(b)加法操作然后重新赋给该位置的data，从而达到累加器的累加效果
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    # 重新设置空间大小并初始化。
    def reset(self):
        self.data = [0.0] * len(self.data)

    # 实现类似数组的取操作
    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 判断net模型是否为深度学习类型，将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()  # 要计算梯度

    # Accumulator(3)创建3个变量：训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)

        # 判断updater是否为优化器
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  # 把梯度设置为0
            l.mean().backward()  # 计算梯度
            updater.step()  # 自更新
        else:
            # 使用定制的优化器和损失函数
            # 自我实现的话，l出来是向量，先求和再求梯度
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度，metric的值由Accumulator得到
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    # num_epochs：训练次数
    for epoch in range(num_epochs):
        # train_epoch_ch3：训练模型，返回准确率和错误度
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)

        # 在测试数据集上评估精度
        test_acc = evaluate_accuracy(net, test_iter)

        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y) # 实际标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1)) 预测标签，取最大化概率
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
