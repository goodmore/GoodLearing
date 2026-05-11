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

y_true=2.5
y_pred=result['y_pred']
a1=result['a1']
z1=result['z1']
x=x_input
print(f"真实值 y_true = {y_true}")
print(f"预测值 y_pred = {y_pred:.4f}")
print(f"初始损失 Loss = {(y_true - y_pred)**2:.4f}")
print("\n--- 开始反向传播计算梯度 ---")
# 1. 计算损失对y_pred的梯度
dL_dy_pred = -2 * (y_true - y_pred)
print(f"梯度 ∂L/∂y_pred = -2*(y_true - y_pred) = {dL_dy_pred:.4f}")
#2.计算损失对w2,b2的梯度
dL_dw2=dL_dy_pred*a1
print(f"梯度∂L/∂w2=∂L/∂y_pred*∂y_pred/∂w2={dL_dw2:.4f}")#//由链式法则可知
dL_db2=dL_dy_pred*1
print(f"梯度∂L/∂b2=∂L/∂y_pred*∂y_pred/∂b2={dL_db2:4f}")
#3.计算隐藏层输出的参数a1
dL_da1=dL_dy_pred*w2
print(f"梯度梯度∂L/∂a1=∂L/∂y_pred*∂y_pred/∂a1={dL_da1:.4f}")
# 4. 计算Sigmoid函数的导数在z1处的值
def sigmoid_derivative(x):
    """Sigmoid函数的导数"""
    s = sigmoid(x) #导数推导法则
    return s * (1 - s)
sigma_prime_z1 = sigmoid_derivative(z1)
print(f"Sigmoid导数 σ'(z1) = σ(z1)*(1-σ(z1)) = {sigma_prime_z1:.4f}")
#5.计算隐藏层参数w1,b1的梯度
"""由前向传播过程可知：
    a1是经过了sigmoid的z1的输出，也就是
    a1=sigmoid(z1)
    #结合前向传播函数公式可知，给出并定义传播过程：
    z1=w1*x+b1
    #所以根据链式法则可知：
    ∂L/∂w1=(∂L/∂y_pred)*(∂y_pred/∂a1)*(∂a1/∂z1)*(∂z1/∂w1)
    ∂L/∂b1=(∂L/∂y_pred)*(∂y_pred/∂a1)*(∂a1/∂z1)*(∂z1/∂b1)
    #并结合之前反向传播计算导数可知，其过程和步骤可以简化：
    ∂L/∂w1=(∂L/∂a1)*(∂a1/∂z1)*(∂z1/∂w1)
    ∂L/∂b1=(∂L/∂a1)*(∂a1/∂z1)*(∂z1/∂b1)
    由sigmoid求导法则可知：
    sigmoid=1/(1+np.exp(-z1)))
    #由前导后不导的分式求导法则可知：
    sigmoid'=-np.exp(-z1)/(1+np.exp(-z1))**2
            =-np.exp(-z1)/(1+np.exp(-z1)*(1+np.exp(-z1))
            =1-(1/(1+np.exp(-z1))))*(1+np.exp(-z1))
            =(1-sigmoid)*sigmoid
    """
dL_dw1=dL_da1*sigmoid_derivative(z1)*x
dL_db1=dL_da1*sigmoid_derivative(z1)

print(f"梯度 ∂L/∂w1 = (∂L/∂a1) * σ'(z1) * x = {dL_dw1:.4f}")
print(f"梯度 ∂L/∂b1 = (∂L/∂a1) * σ'(z1) * 1 = {dL_db1:.4f}")

