from linecache import cache

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

def relu(x):
    return np.maximum(0, x)

#初始化简单神经网络
def initialize_network(input_size,hidden_size,output_size):
    """
        初始化网络权重和偏置。
        参数:
            input_size: 输入层神经元数
            hidden_size: 隐藏层神经元数
            output_size: 输出层神经元数
        返回:
            network: 包含各层参数的字典
        """
    np.random.seed(42)
    network={}
    # 初始化 输入层->隐藏层 的参数
    # 权重矩阵形状: (下一层神经元数， 上一层神经元数)
    network['w1']=np.random.randn(input_size,hidden_size)*0.01
    network['b1']=np.zeros((1, hidden_size))#偏置为列向量，其列参数为1
    # 初始化 隐藏层->输出层 的参数
    network['w2']=np.random.randn(hidden_size,output_size)*0.01
    network['b2']=np.zeros((1, output_size))
    return network

#定义前向传播函数
def forward_propagation(network,x):
    """
        执行前向传播，计算网络输出。
        参数:
            network: 包含权重和偏置的字典
            X: 输入数据，形状为 (特征数, 样本数)
        返回:
            y_pred: 网络预测输出
            cache: 缓存中间结果（用于后续的反向传播）
        """
    #获取之前定义的参数
    w1,b1,w2,b2=network['w1'],network['b1'],network['w2'],network['b2']

    #第一层计算：输入层->隐藏层
    z1=np.dot(x,w1)+b1
    #计算第一层激活函数结果
    a1=relu(z1)
    #第二层计算：隐藏层->输出层
    z2=np.dot(a1,w2)+b2
    #计算第二层激活函数结果
    a2=relu(z2)

    #缓存中间参数或者输出结果
    cahce={'z1':z1,'a1':a1,'a2':a2,'z2':z2}

    #返回第二层激活函数的结果和中间层参数的缓存
    return a2,cache

#开始主程序参数定义
#1.初始化神经网络基础参数，输入层数：50，隐藏层数：100，输出层数：10
input_size=2
hidden_size=3
output_size=1

#2.初始化网络
first_network=initialize_network(input_size,hidden_size,output_size)
print("权重w1的形状(隐藏层 x 输入层):",first_network['w1'].shape)
print("偏执b1的形状：",first_network['b1'].shape)
print("权重w2的形状(隐藏层 x 输出层)：",first_network['w2'].shape)

#3.创建样本输入数据
x_sample=np.array([[1,0.5],[1,4]])
print("\n输入数据：",x_sample.T)

#4.执行前向传播
y_pred,cache=forward_propagation(first_network,x_sample)
print("\n神经网络预测输出(a2)：",y_pred)
