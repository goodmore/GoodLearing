import numpy as np


def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


# 初始化网络参数（通常随机初始化，这里为演示指定值）
w1, b1 = 2.0, -1.0  # 隐藏层参数
w2, b2 = 1.5, 0.5  # 输出层参数


def forward_pass(x):
    """执行一次前向传播"""
    # 隐藏层计算
    z1 = w1 * x + b1
    a1 = sigmoid(z1)  # 应用激活函数

    # 输出层计算
    y_pred = w2 * a1 + b2  # 线性输出

    # 返回中间结果和最终预测，便于后续理解
    return {'z1': z1, 'a1': a1, 'y_pred': y_pred}


# 假设房屋面积为 3（单位：百平方米）
x_input = 3.0
result = forward_pass(x_input)
print(f"输入 x = {x_input}")
print(f"隐藏层线性输出 z1 = w1*x + b1 = {result['z1']:.4f}")
print(f"隐藏层激活输出 a1 = sigmoid(z1) = {result['a1']:.4f}")
print(f"最终预测房价 y_pred = w2*a1 + b2 = {result['y_pred']:.4f}")