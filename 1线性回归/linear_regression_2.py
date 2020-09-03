'''
python实现线性回归
'''

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([2.1, 3.8, 5.9, 8.0, 10.2])

w = 1.0
b = 1.0
learning_rate = 0.001

#预测得到正确结果
def forward(x):
    return w * x + b

#计算损失函数，使用平方损失
def loss(x, y):
    y_pred = forward(x)                 #计算出预测结果
    return (y_pred - y) ** 2

#梯度下降法，计算梯度
def gradient(x, y):
    w_grad = 0
    b_grad = 0
    w_grad += 2 * x * (w * x + b - y)
    b_grad += 2 * (w * x + b - y)
    return w_grad, b_grad

for epoch in range(5000):             #共迭代5000轮
    i = 0
    while i < x_train.shape[0]:
        Loss = loss(x_train[i], y_train[i])
        w_gradient, b_gradient = gradient(x_train[i], y_train[i])
        w = w - learning_rate * w_gradient  #更新权重w
        b = b - learning_rate * b_gradient  #更新权重b
        i +=1


print("最后预测结果为：",forward(7.3))


plt.ylabel("y")
plt.xlabel("x")
plt.scatter(x_train, y_train)
x = np.linspace(0,10,100)
y = forward(x)
plt.plot(x,y,color='red')
plt.show()