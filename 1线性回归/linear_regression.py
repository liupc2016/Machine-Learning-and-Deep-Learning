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
def loss(x_train, y_train):
    LOSS = 0
    i = 0
    while i < x_train.shape[0]:    #样本数
        y_pred = forward(x_train[i])    #计算出预测结果
        LOSS += (y_pred - y_train[i]) ** 2
        i = i+1
    return LOSS/x_train.shape[0]

#梯度下降法，计算梯度
def gradient(x_train, y_train):
    w_sum = 0
    b_sum = 0
    i = 0
    while i< x_train.shape[0]:
        w_sum += 2 * x_train[i] * (w * x_train[i] + b - y_train[i])
        b_sum += 2 * (w * x_train[i] + b - y_train[i])
        i = i+1
    return w_sum/x_train.shape[0], b_sum/x_train.shape[0]


mse_list = []      #为了绘图用
epoch_list = []    #为了绘图用

for epoch in range(5000):             #共迭代5000轮
    LOSS = loss(x_train, y_train)      #每一轮首先计算损失(为了绘图用)
    w_gradient, b_gradient = gradient(x_train, y_train)
    w = w - learning_rate * w_gradient  #更新权重w
    b = b - learning_rate * b_gradient  #更新权重b

    print('Epoch:',epoch, " w:", w, " b:",b, " loss:",LOSS)
    mse_list.append(LOSS)
    epoch_list.append(epoch)

print("最后预测结果为：",forward(7.3))

plt.subplot(121)
plt.ylabel("y")
plt.xlabel("x")
plt.scatter(x_train, y_train)
x = np.linspace(0,10,100)
y = forward(x)
plt.plot(x,y,color='red')

plt.subplot(122)
plt.plot(epoch_list, mse_list)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()