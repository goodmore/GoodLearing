import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x=3*np.random.rand(100,1)
y=3+4*x+np.random.randn(100,1)

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression')
plt.show()

w=0
b=0
learning_rate=0.01 #//使用均方误差来作为损失函数
n=1000

for i in range(n):
    y_pred=w*x+b
    dw=-(2/len(x))*np.sum(x*(y-y_pred))
    db=-(2/len(x))*np.sum(y-y_pred)
    w=w-learning_rate*dw
    b=b-learning_rate*db

print(f"手动实现的参数:{w}")
print(f"手动实现的截距:{b}")

y_pred_manual=w*x+b
plt.scatter(x,y)
plt.plot(x,y_pred_manual,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Manual Gradient Descent Fit')
plt.show()
