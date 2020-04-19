#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import sys
sys.path.append('../DQN')
sys.path.append('../GridWorldv1')

"""
DDPG中:
Critic网络充当评论家:
--- 它估计个体在当前状态下的价值以指导策略产生行为
Actor网络:
--- 根据当前状态生成具体的行为
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" 
使用PyTorch库中的神经网络来构建这两个近似函数
"""

# 为了增加模型的收敛性，使用一种更加有效的网络参数初始化方式
def fanin_init(size,fanin=None):
    """https://arxiv.org/abs/1502.01852"""
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v,v) # 从-v到v的均匀分布
    return x.type(torch.FloatTensor)

# Critic网络：
# input: 个体观测的特征数以及行为的特征数
# output: 状态行为对的Q值
# 3个hidden layers
# 处理状态的隐藏层和行为的隐藏层先分开运算，通过最后一个隐藏层全连接
# 在一起输出状态行为对价值。
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        """
        构建评论家模型
        :param state_dim: 状态的特征的数量(int)
        :param action_dim:  行为的特征的数量(int)
        """
        super(Critic,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 状态线
        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size()) #初始化
        self.fcs2 = nn.Linear(256,128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        # 动作线
        self.fca1 = nn.Linear(action_dim,128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        # 状态+行为的联合线性变换
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        # 状态+行为的联合线性变换
        self.fc3 = nn.Linear(128,1)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

    def forward(self,state,action):
        """
        前向传播，输入state和action的特征，给出Q值
        :param state: torch Tensor [n,state_dim] 状态的特征表示
        :param action: torch Tensor [n,action_dim] 行为的特征表示
        :return: Q(s,a) torch Tensor [n,1]
        """
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)
        action = action.type(torch.FloatTensor)

        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action))

        # 将状态和行为链接起来，使用第二种近似架构(s,a) -> Q(s,a)
        x = torch.cat((s2,a1),dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#
# Actor网络
# 输入时个体观测的特征数
# 输出每一个行为特征具体的值
EPS = 0.003 # epsilon
class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,action_lim):
        """
        演员模型
        :param state_dim: 状态特征数
        :param action_dim: 行为特征数
        :param action_lim: 行为值范围[-action_lim,action_lim]
        """
        super(Actor,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(self.state_dim,256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64,self.action_dim)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())

    def forward(self,state):
        """前向"""
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor) # float类型张量
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x))
        action = action * self.action_lim # 输出范围更改
        return action

# OU噪声
# Ornstein-Unlenbeck
# 用于连续行为空间下实现探索，使其在确切的行为周围实现一定范围的探索
class OrnsteinUhlenbeckActionNoise:
    def __init__(self,action_dim,mu=0,theta=0.15,sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim)*self.mu

    def reset(self):
        self.X = np.ones(self.action_dim)*self.mu # 初始化

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma*np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


# if __name__ == '__main__':
    # action_dim = 22
    # OU_noise = OrnsteinUhlenbeckActionNoise(action_dim)
    # print(OU_noise.sample()) # 用来加在选择好的action上
