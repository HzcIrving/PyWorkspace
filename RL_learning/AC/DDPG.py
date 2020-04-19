#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
DDPG实现
继承Agent基类
"""

import sys
sys.path.append('../GridWorldv1')

from AC import *
from puckworld_continuous import *

from random import random,choice
from gym import Env,spaces
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from EpisodeManage import Transition,Experience
from Agent import *

# ---- 软更新与硬更新 ----
def soft_update(target,source,tau):
    """
    y = tau * x + (1 - tau) * y
    将source网络(x)参数软更新至target网络(y)参数
    """
    for target_param,param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(
            target_param.data*(1.0-tau) + param.data*tau
        )

def hard_update(target,source):
    """硬更新"""
    for target_param,param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(param.data)

# ---- DDPG实现 ----
# Agent包括的能力：
# -- 1. policy : 当前默认策略，均一随机策略 本次需重写
# -- 2. perform_policy : 执行上面的策略，得到action
# -- 3. act : 执行action,得到s1,reward...
# -- 4. Learning_method : 学习的方法
# -- 5. learning : 学习过程
# -- 6. sample : 随机采样
# -- 7. total_trans : Experience里记载的总的状态转换数量
# -- 8. last_episode_detail : 打印最后一个状态序列信息
class DDPGAgent(Agent):
    """
    使用Actor-Critic算法结合深度学习的个体
    """
    def __init__(self,env:Env=None,
                      capacity = 2e6,
                      batch_size = 128,
                      action_lim = 1,
                      learning_rate = 0.001,
                      gamma = 0.999,
                      epochs = 2):
        if env is None:
            raise "没环境啊大哥..."
        super(DDPGAgent,self).__init__(env,capacity)

        # 任务基本信息
        self.state_dim = env.observation_space.shape[0] # 状态连续
        self.action_dim = env.action_space.shape[0] # 行为连续

        self.action_lim = action_lim
        self.batch_size = batch_size # 批学习一次状态转换数量
        self.learning_rate = learning_rate
        self.gamma = 0.999  #衰减因子
        self.epochs = epochs #一批状态转换学习的次数
        self.tau = 0.001 # 软更新系数
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        # Actor
        self.actor = Actor(self.state_dim,self.action_dim,self.action_lim)
        # Target Actor
        self.target_actor = Actor(self.state_dim,self.action_dim,self.action_lim)

        # Critic
        self.critic = Critic(self.state_dim,self.action_dim)
        # Target Critic
        self.target_critic = Critic(self.state_dim,self.action_dim)

        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 self.learning_rate)

        # 初始化第一次直接硬拷贝
        hard_update(self.target_actor,self.actor)
        hard_update(self.target_critic,self.critic)

    # exploitation & exploration
    def get_exploitation_action(self,state):
        """
        代替原类中的policy,因为是连续行为，放弃Agent类的policy方法，
        转而声明下面两个新方法分别实现确定性策略中的探索和利用。
        input: state (numpy)
        output: action (numpy)
        """

        # detach: 截断反向传播的梯度流
        # 返回一个新的从当前图中分离的Variable
        # 返回的Variable永远不会需要梯度
        # 如果被detach的Variable volatile=True，那么detach出来的volatile也为True
        # 返回的Variable和被detach的Variable指向同一个tensor
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self,state):
        """得到给定状态下，根据Actor网络计算出的带有噪声的行为
        即 a = mu(s_t|theta)+N_t中求a_t的过程"""
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample())*self.action_lim
        new_action = new_action.clip(min=-1*self.action_lim,
                                     max=self.action_lim)
        return new_action

    # 同DQN，DDPG算法也是基于经历回放，且参数的更新均通过训练从经历随机得到
    # 的多个状态转换而得到
    def _learn_from_memory(self):
        """从记忆中学习，更新两个网络参数"""

        # 随机采样batch_size个记忆里的Transition
        trans_pieces = self.sample(self.batch_size)
        s0 = np.vstack([x.s0 for x in trans_pieces])
        a0 = np.array([x.a0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        # is_done =
        s1 = np.vstack([x.s1 for x in trans_pieces])

        # -----------优化critic网络参数----------------
        a1 = self.target_actor.forward(s1).detach()
        next_val = torch.squeeze(self.target_critic.forward(s1,a1).detach()) # Q_target

        r1 = torch.from_numpy(r1)
        y_expected = r1 + self.gamma*next_val
        y_expected = y_expected.type(torch.FloatTensor)

        a0 = torch.from_numpy(a0) # 从numpy->tensor
        y_predicted = torch.squeeze(self.critic.forward(s0,a0))

        # critic loss
        loss_critic = F.smooth_l1_loss(y_predicted,y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward() # BP
        self.critic_optimizer.step()

        # ----------优化actor网络参数 --------------------
        # 目标：使Q值增大
        pred_a0 = self.actor.forward(s0)

        # 策略梯度，求max，应该梯度上升
        loss_actor = -1 * torch.sum(self.critic.forward(s0,pred_a0))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # ----------soft update target nets params -----------
        soft_update(self.target_actor,self.actor,self.tau)
        soft_update(self.target_critic,self.critic,self.tau)

        return (loss_critic.item(),loss_actor.item())

    # 学习方法 个体与环境实际交互，实现一个完整的序列状态
    # 注意返回的是：
    # 1. time_in_episode : 一个完整的序列状态中包含时间步数
    # 2. total_reward : 一个episode完成时的奖励
    # 3. loss_critic : 一个episode的平均critic loss
    # 4. loss_actor : 一个episode的平均actor loss
    def learning_method(self,display=False,explore=True):
        """一个episode的过程，当is_done完成，即为一个完整的episode"""
        # 1. 初始化
        self.state = np.float64(self.env.reset())
        time_in_episode = 0
        total_reward = 0
        is_done = False
        loss_critic = 0
        loss_actor = 0

        # 2. 一个epsiode的交互
        while not is_done:
            s0 = self.state
            if explore:
                a0 = self.get_exploration_action(s0)
            else:
                a0 = self.actor.forward(s0).detach().data.numpy()

            s1,r1,is_done,info,total_reward = self.act(a0)

            if display:
                self.env.render()

            if self.total_trans > self.batch_size:
                loss_c,loss_a = self._learn_from_memory()
                loss_critic += loss_c
                loss_actor += loss_a

            time_in_episode += 1 # 用于计数时间步数

        # 3. 计算一个episode中的平均loss
        loss_critic /= time_in_episode
        loss_actor /= time_in_episode

        if display:
            print("{}".format(self.experience.last_episode()))

        return time_in_episode,total_reward,loss_critic,loss_actor


    # 学习过程
    def learning(self,max_episode_num=800,display=False,explore=True):
        # 1.初始化
        total_time = 0
        total_times = []
        episode_reward = 0
        episode_rewards = []
        num_episode = 0
        num_episodes = []

        # 2.学他
        for i in tqdm(range(max_episode_num)):
            time_in_episode,episode_reward,loss_critic,loss_actor = \
                    self.learning_method(display,explore)

            total_time += time_in_episode
            total_times.append(total_time)

            num_episode += 1
            num_episodes.append(num_episode)

            episode_rewards.append(episode_reward)

            print("episode:{:3}: loss critic:{:4.3f}, loss_actor:{:4.3f}".\
                  format(num_episode-1,loss_critic,loss_actor))

            if explore and num_episode%200 == 0:
                self.save_models(num_episode)

        return total_times,episode_rewards,num_episodes

    def save_models(self,episode_count):
        """保存"""
        torch.save(self.target_actor.state_dict(),'./Models/',+str(
            episode_count)+'_actor.pt'
        )
        torch.save(self.target_critic.state_dict(),'./Models/',+str(
            episode_count)+'_critic.pt'
        )
        print("模型成功保存!")

    def load_models(self,episode):
        self.actor.load_state_dict(torch.load("./Models/"+str(episode)+'_actor.pt'))
        self.critic.load_state_dict(torch.load("./Models/"+str(episode)+'_critic.pt'))
        hard_update(self.target_actor,self.actor)
        hard_update(self.target_critic,self.critic)
        print("模型读取成功!")


if __name__ == '__main__':

    env = PuckWorldEnv()
    agent = DDPGAgent(env)

    # 1.启动学习过程
    data = agent.learning(max_episode_num=200,display=False)




