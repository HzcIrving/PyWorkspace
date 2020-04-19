#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
近似函数设计
输入s,a
输出Q值

环境：连续状态、离散行为

一个隐藏层 32个神经元
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy

class NetApproximator(nn.Module):
    def __init__(self,input_dim=1,output_dim=1,hidden_dim=32):
        """
        近似价值函数
        :param input_dim: 输入层特征数
        :param output_dim:  输出层特征数
        :param hidden_dim:  隐层神经元数
        """
        super(NetApproximator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim,output_dim)
        # 非线性整合功能Pytorch本身提供,不需要参数

    def forward(self,x):
        """前向运算"""
        x = self._prepare_data(x) # 需要对描述状态的输入参数x做一定的处理
        h_relu = F.relu(self.linear1(x)) # 非线性整合函数relu
        y_pred = self.linear2(h_relu) # 网络输出
        return y_pred

    def _prepare_data(self,x,requires_grad=False):
        """将numpy格式的数据转化为Torch的Variable"""
        if isinstance(x,np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x,int): # 可接受单个数据
            x = torch.Tensor([[x]]) # 转为Tensor
        x.requires_grad = requires_grad # 是否打开自动梯度功能
        x = x.float()
        if x.data.dim() == 1:
            x = x.unsqueeze(0) # torch的nn接受的输入至少都是2维的
        return x

    def fit(self,x,y,criterion=None,optimizer=None,epochs=1,learning_rate=1e-4):
        """通过训练更新网络参数来拟合给定的输入x和y"""
        if criterion is None:
            criterion = torch.nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)

        if epochs < 1:
            epochs = 1 # 对参数给定的数据训练的测试

        y = self._prepare_data(y,requires_grad=False)

        for t in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred,y) # loss

            # BP
            optimizer.zero_grad() # 梯度重置
            loss.backward() # BP  自动计算相应节点的梯度
            optimizer.step() #更新权重

        return loss # 返回本次训练最后一个epoch的损失

    """辅助函数"""
    def __call__(self,x):
        """该类的对象接受参数直接返回结果"""
        y_pred = self.forward(x)
        return y_pred.data.numpy() #test用

    def clone(self):
        """返回当前模型的深度拷贝对象"""
        return copy.deepcopy(self)








