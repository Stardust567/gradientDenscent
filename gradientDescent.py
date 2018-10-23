import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random

w_init = 4.2
b_init = -150

def f(x):
    return w_init * x + b_init

def initDate():
    x_data = []
    y_data = []
    for i in range(10):
        x = random.randint(-99,99)
        x_data.append(x)
        y_data.append(f(x)+f(x)/10*random.randint(-1,1))
        # 给数据加点噪声，使过程更加真实一点
    return x_data,y_data

def exhaustion(x_data,y_data):
    # 通过穷举法试探出使均方误差最小的一组b&w
    x = np.arange(-abs(1.5*b_init),abs(1.5*b_init),abs(b_init)/20) #bias
    y = np.arange(-abs(1.5*w_init),abs(1.5*w_init),abs(w_init)/20) #weight
    Z = np.zeros((len(x),len(y)))
    #zeros(shape, dtype=float, order='C')return一个给定形状和类型的用0填充的数组
    X,Y = np.meshgrid(x,y)
    #return两个矩阵,X的行向量是向量x的简单复制,Y的列向量是向量y的简单复制
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    for i in range(len(x)):
        for j in range(len(y)):
            b = x[i]
            w = y[j]
            Z[j][i] = ((y_data - b - w * x_data)**2).sum()/len(x_data)
            # Z[j][i]值最小的那一组x[i].y[j]就是我们所期望的使loss funcion值最小的b，w
           
    return X,Y,Z

def gradientDescent(x_data,y_data):
    #initial:
    b = 0
    w = 0
    lr = 1 # learning rate
    iteration = 100000

    b_history = [b]
    w_history = [w]
    # 我们实际用gandient descent求出的b&w

    lr_b = 0
    lr_w = 0

    for i in range(iteration):
    #不断向偏导为0的驻点靠拢，以获取均方误差最小的一组解w、b
        
        x = np.array(x_data)
        y = np.array(y_data) 
        b_grad = -2.0*(y - b - w*x) # 梯度（偏导）
        w_grad = -2.0*(y - b - w*x)*x
        b_grad = b_grad.sum()/len(x)
        w_grad = w_grad.sum()/len(x)

        lr_b += b_grad ** 2
        lr_w += w_grad ** 2

        #update parameters: 
        b -= lr/np.sqrt(lr_b) * b_grad
        w -= lr/np.sqrt(lr_w) * w_grad
    
        #store parameters for plotting:
        b_history.append(b)
        w_history.append(w)

    return b_history,w_history

def getPicture(b_history,w_history,X,Y,Z):
    plt.figure('gradient_descent')
    plt.contourf(X,Y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
    plt.plot([b_init],[w_init],'x',ms=12,markeredgewidth=3,color='orange')
    plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
    plt.xlim(-abs(1.5*b_init),abs(1.5*b_init))
    plt.ylim(-abs(1.5*w_init),abs(1.5*w_init))
    plt.xlabel(r'$b$',fontsize=16)
    plt.ylabel(r'$w$',fontsize=16)

def get3D(b_history,w_history,X,Y,Z):
    #为了画好看的三维图并考虑到memory error强行又弄了组数据
    b_h = b_history[::1000]
    w_h = w_history[::1000]
    B,W = np.meshgrid(b_h,w_h)
    Q = np.zeros(len(b_h))
    for i in range(len(b_h)):    
        for n in range(len(x_data)):
            Q[i] = Q[i] + (y_data[n] - b_h[i] - w_h[i] * x_data[n])**2 
        Q[i] = Q[i]/len(x_data) # 均方误差
        
    ax = Axes3D(plt.figure('三维图'))
    ax.plot_surface(X,Y,Z,cmap = 'rainbow')
    ax.plot([b_init],[w_init],'x',ms=12,markeredgewidth=3,color = 'orange')
    ax.plot(b_h,w_h,Q,'o-',ms=3,lw=1.5,color='black')
    ax.set_xlabel('--b--')  
    ax.set_ylabel('--w--')
    ax.set_zlabel('--z--')
    ax.set_title('3D')

def initPicture(b,w):
    plt.figure('initial')
    x = np.linspace(-99,99) 
    y_init = f(x) # 所求目标函数的函数值
    y_grad = w*x+b # 梯度下降求出的函数所对应的函数值
    plt.plot(x,y_init,'.')
    plt.plot(x,y_grad)

if __name__ == '__main__':
    x_data,y_data = initDate() # 获取实验数据
    X,Y,Z = exhaustion(x_data,y_data) # X&Y为网格图，Z为其函数值
    b_history,w_history = gradientDescent(x_data,y_data) # 得出梯度下降所求的b&w
    b = b_history[-1];print(b)
    w = w_history[-1];print(w)
    
    getPicture(b_history,w_history,X,Y,Z) # 绘制图像
    get3D(b_history,w_history,X,Y,Z)
    initPicture(b,w)
    plt.show()



