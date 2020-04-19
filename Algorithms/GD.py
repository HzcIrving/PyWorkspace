#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
gradient descent
梯度下降算法
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

"""
二维梯度下降算法
"""

w1 = np.arange(-2,8.02,0.02)
w2 = np.arange(-4,6.02,0.02)
w1,w2 = np.meshgrid(w1,w2)

w1_true = 3  # True label for w1
w2_true = 1  # True label for w2
w1_0,w2_0 = -1,-3

def cost_func(w1,w2):
    return np.power(5*(w1-w1_true),2) + np.power(4*(w2-w2_true),2)

J = cost_func(w1,w2)

def grad_w1(w1):
    # 对w1求导
    return 5*2*(w1-3)

def grad_w2(w2):
    # 对w2求导
    return 4*2*(w2-1)

def cut_plain(w1,w2):
    k1,k2 = grad_w1(w1_0),grad_w2(w2_0)
    return k1*(w1-w1_0) + k2*(w2-w2_0) + cost_func(w1_0,w2_0) + 300


fig = plt.figure(figsize=(10,5))
fig.suptitle(r'3D surface of $J(w_1,w_2)$ and gradient descent')
ax0 = fig.add_subplot(1,2,1,projection='3d')

ax0.plot_surface(w1,w2,J,rstride=8,cstride=8,alpha=1,cmap=cm.rainbow)
ax0.contourf(w1,w2,J,zdir='z',offset=-100,cmap=cm.rainbow)

plt.show()
