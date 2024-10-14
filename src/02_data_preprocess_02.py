# %matplotlib inline
import pandas as pd
import torch

data = pd.read_csv('../data/house_tiny.csv')
print(data)

inputs, outputs = data.iloc[:,0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())
# print(inputs)

###处理缺失值

# 只对数值型列进行填充
numeric_cols = inputs.select_dtypes(include='number')
inputs[numeric_cols.columns] = inputs[numeric_cols.columns].fillna(numeric_cols.mean())

print(inputs)
# 将非数值型的数据转化为数值型
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# pd.get_dummies 是 Pandas 中用于将分类数据（如字符串类型或类别类型数据）转换为
# 一组虚拟变量（dummy variables）或二进制编码的函数。
# 这种转换有助于将非数值型的数据转化为数值型，以便在机器学习模型中使用

# df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']})
# dummies = pd.get_dummies(df, columns=['Color'])
# print(dummies)

## 转换为张量格式
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(y)