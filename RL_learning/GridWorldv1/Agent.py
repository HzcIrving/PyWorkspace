#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
不同学习能力的Agent
1.Sarsa
2.Sarsa(lambda)
3.Q-Learning
"""

from DifferentGridEnv import *
from GridWorldv1 import *
from EpisodeManage import *
from random import random
from gym import Env
import gym
from Toolutils import *

# 一同通用Agent应该具备什么能力?
## 1. 对环境对象的应用
## 2. 状态和行为空间
## 3. 与环境交互产生的经历
## 4. 当前状态
## 5. 遵循策略产生行为
## 6. 执行一个行为与环境交互
## 7. 采用什么学习方法

class Agent(object):
    """个体基类，没有学习能力"""
    def __init__(self,Env=None,capacity=10000):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = Env # 建立对环境对象的引用
        self.obs_space = self.env.observation_space if self.env is not None else None
        self.action_space = self.env.action_space if self.env is not None else None
        if type(self.obs_space) in [gym.spaces.Discrete]:
            self.S = [str(i) for i in range(self.obs_space.n)]
            self.A = [str(i) for i in range(self.action_space.n)]
        else:
            self.S,self.A = None,None
        self.experience = Experience(capacity=capacity)

        # 有一个变量记录agent当前的state，注意对当前变量的更新与维护
        self.state = None # current state

    def policy(self,A,s=None,Q=None,epsilon=None):
        """当前默认策略，均一随机策略"""
        return random.sample(self.A,k=1)[0]

    def perform_policy(self,s,Q=None,epsilon=0.05):
        """得到执行的policy"""
        action = self.policy(self.A,s,Q,epsilon)
        return int(action)

    def act(self,a0):
        s0 = self.state # 获得当前的state
        s1,r1,is_done,info = self.env.step(a0)
        trans = Transition(s0,a0,r1,is_done,s1) # 存储trans
        total_reward = self.experience.push(trans) # 加入经验池
        self.state = s1
        return s1,r1,is_done,info,total_reward

    def learning_method(self,lambda_=0.9,gamma=0.9,alpha=0.5,epsilon=0.2,
            display=False,current_episode=None):
        """Agent使用的是默认的学习方法，即不具备学习能力"""
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0,epsilon)
        time_in_episode,total_reward = 0,0
        is_done = False #默认是False
        while not is_done:
            s1,r1,is_done,info,total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1,epsilon)
            s0,a0 = s1,a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode,total_reward

    def learning(self,lambda_=0.9,epsilon=None,decay_epsilon=True,gamma=0.9,
                 alpha=0.1,max_episode_num=800,display=False):
        total_time,episode,num_episode=0,0,0 # 计数器
        total_times,episode_rewards,num_episodes=[],[],[]
        for i in tqdm(range(max_episode_num)):
            if epsilon is None:
                epsilon = 1e-10
            elif decay_epsilon:
                epsilon = 1.0/(1+num_episode)
            time_in_episode,episode_reward = self.learning_method(lambda_=lambda_,
                                    gamma=gamma,alpha=alpha,epsilon=epsilon,display=display,current_episode=i)
            total_time += time_in_episode
            num_episode+=1
            total_times.append(total_time) # 当前epsiode总时间
            episode_rewards.append(episode_reward) #当前episode的总奖励
            num_episodes.append(num_episode) # 当前episode的总数
            total_time=0
        return total_times,episode_rewards,num_episodes

    def sample(self,batch_size=64):
        """随机采样"""
        return self.experience.sample(batch_size)

    @property
    def total_trans(self):
        """Experience里记载的总的状态转换数量"""
        return self.experience.total_trans

    def last_episode_detail(self):
        self.experience.last_episode.print_detail()

# if __name__ == "__main__":
#     env = WindyGridWorld()
#     # env = CliffWalk2()
#     env.reset()
#     env.render()
#
#     agent = Agent(env,capacity=10000)
#     data = agent.learning(max_episode_num=180,display=False)
#     learning_curve(data,2,0,title="learning curve",x_name="Episodes",y_name="Time steps")


