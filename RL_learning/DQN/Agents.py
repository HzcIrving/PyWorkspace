#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import sys
sys.path.append('../GridWorldv1')

from Agent import *
from DQN_netv2 import *
import gym
from Puckworld import PuckWorldEnv
from Toolutils import  learning_curve

from random import random

import warnings
warnings.filterwarnings("ignore")

"""
需要继承Agent的基本属性：
# 一同通用Agent应该具备什么能力?
## 1. 对环境对象的应用
## 2. 状态和行为空间
## 3. 与环境交互产生的经历
## 4. 当前状态
## 5. 遵循策略产生行为
## 6. 执行一个行为与环境交互
## 7. 采用什么学习方法
"""

class DQNAgnet(Agent):
    """使用近似的价值函数实现Q的学习个体"""
    def __init__(self,env:Env=None,capacity=20000,hidden_dim:int=32,
                 batch_size=128,epochs=2):
        if env is None:
            raise "Are u kidding me,there isn't a valiable env"
        super(DQNAgnet, self).__init__(env,capacity)
        self.input_dim = env.observation_space.shape[0] # 状态连续
        self.output_dim = env.action_space.n # 离散行为,int表示
        self.hidden_dim = hidden_dim

        # ----------------------------------------------
        # 行为网络，策略产生实际交互行为的依据
        # 输入s，输出Q(s,a)
        # 即当前状态，采取各个action的价值进行评估近似
        self.behavior_Q = NetApproximator(
                        input_dim = self.input_dim,
                        output_dim = self.output_dim,
                        hidden_dim = self.hidden_dim
        )
        # Target网络，初始时从Q网络copy，参数一致，不定期更新
        # 目标价值网络，根据状态和行为得到目标价值，计算代价的依据
        self.target_Q = self.behavior_Q.clone()
        # ----------------------------------------------

        self.batch_size = batch_size
        self.epochs = epochs # 一次学习对mini_batch个状态转换训练的次数

    def _update_target_Q(self):
        """更新目标价值网络"""
        self.target_Q = self.behavior_Q.clone()

    def policy(self,A,s=None,Q=None,epsilon=None):
        """依据，亦然是epsilon-greedy策略"""
        Q_s= self.behavior_Q(s) # 基于NetApproximator实现了__call__方法
        rand_value = random() # 生成0与1之间的随机数
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(Q_s))

    def _learn_from_memory(self,gamma,learning_rate):
        """从记忆库中学习"""
        trans_pieces = self.sample(self.batch_size) # 随机获取记忆里的状态转换
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        # --------training....---------
        X_batch = states_0
        y_batch = self.target_Q(states_0)

        # # ---------DQN---------------------
        # # 求Target Q
        # # 若is_done : Q_target = reward1
        # Q_target = reward_1 + gamma * np.max(self.target_Q(states_1),axis=1)*(~ is_done)


        # ------DDQN----------------------
        # 行为a'从行为价值网络中得到
        a_prime = np.argmax(self.behavior_Q(states_1),axis=1).reshape(-1)
        # (s',a')的价值从目标价值网络中得到
        Q_states_1 = self.target_Q(states_1)
        temp_Q  = Q_states_1[np.arange(len(Q_states_1)),a_prime]
        #(s,a)的目标价值依据bellman:
        Q_target = reward_1 + gamma * temp_Q * (~ is_done)


        # DQN
        # y_batch对于着前len(X_batch)长度的Q_target
        y_batch[np.arange(len(X_batch)),actions_0] = Q_target

        # 训练行为价值网络,更新其参数
        loss = self.behavior_Q.fit(x=X_batch,y=y_batch,
                        learning_rate=learning_rate,
                        epochs=self.epochs)

        mean_loss = loss.sum().item()/self.batch_size

        self._update_target_Q()

        return mean_loss # 这一batch_size的平均loss

    def learning_method(self,lambda_=0.9,gamma=0.9,alpha=0.5,epsilon=0.2,
            display=False,current_episode=None):
        """DQN学习方法：
        核心：根据当前状态特征S_0依据行为策略生成一个与环境交互的行为A0;
        交互后观察环境，得到奖励R_1,下一状态特征S_1,判断状态序列是否结束，
        随后将得到的Transition纳入记忆中;
        在每一个时间步内，只要记忆中的Transitions足够多,随机从中提取一定量
        的状态转换基于记忆的学习，实现网络参数更新
        """
        self.state = self.env.reset()
        s0 = self.state

        if current_episode > 50 and display==True:
            self.env.render()

        timesteps_in_episode,total_reward = 0,0
        is_done = False
        loss = 0

        while not is_done:
            s0 = self.state
            a0 = self.perform_policy(s0,epsilon)
            s1,r1,is_done,info,total_reword = self.act(a0) # 与环境交互
            if current_episode > 50 and display == True:
                self.env.render()

            if self.total_trans > self.batch_size:
                # batch_size里的平均loss
                loss += self._learn_from_memory(gamma,alpha)
            timesteps_in_episode += 1

        loss/=timesteps_in_episode

        if display:
            print("\nEpsilon:{:3.2f},loss:{:3.6f},{}".format(epsilon,
                                            loss,self.experience.last_episode))
        return timesteps_in_episode,total_reward


if __name__ == "__main__":
    env = PuckWorldEnv()
    agent = DQNAgnet(env)

    data = agent.learning(gamma=0.99,
                    epsilon=1,
                    decay_epsilon=True,
                    alpha=1e-2, #学习率
                    max_episode_num=100,
                    display=True)

    learning_curve(data,2,1,title="DQN Learning Curve of PuckWorls",
                   x_name="episodes",y_name="rewards of episode")







