import numpy as np
import matplotlib.pyplot as plt

# 定义 x 的范围
x = np.linspace(0.1, 10, 400)

# 计算对数函数值
y_natural_log = np.log(x)  # 自然对数
y_common_log = np.log10(x)  # 常用对数

# 绘制对数曲线
plt.figure(figsize=(8, 6))

plt.plot(x, y_natural_log, label='y = ln(x)', color='b')
plt.plot(x, y_common_log, label='y = log10(x)', color='r')

# 设置图形的标签和标题
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logarithmic Function Curves')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
