#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
基于S.A.R.S.A算法
"""

from Agent import Agent
from gym import Env
import gym
from Toolutils import *
from DifferentGridEnv import *

class SarsaAgent(Agent):
    def __init__(self,env:Env,capacity:int=20000):
        super(SarsaAgent,self).__init__(env,capacity)
        self.Q = {} # Q字典，存储行为价值(Q Table)

    def policy(self,A,s=None,Q=None,epsilon=None):
        """使用epsilon-贪婪策略"""
        return epsilon_greedy_policy(A,s,Q,epsilon)

    def learning_method(self,gamma=0.9,alpha=0.1,epsilon=1e-5,display=False,
                lambda_=None,current_episode=None):
        self.state = self.env.reset()
        s0 = self.state
        if current_episode > 200:
        # if display:
            self.env.render()
        a0 = self.perform_policy(s0,self.Q,epsilon) # epsilon-greedy策略选择action

        timestep_in_episode,total_reward = 0,0
        is_done = False
        while not is_done:
            s1,r1,is_done,info,total_reward = self.act(a0)
            if current_episode > 200 and display==True:
                self.env.render()
            a1 = self.perform_policy(s1,self.Q,epsilon)

            # update Q
            old_q = get_dict(self.Q,s0,a0)
            q_prime = get_dict(self.Q,s1,a1)
            td_target = r1 + gamma * q_prime
            new_q = old_q + alpha*(td_target-old_q)
            set_dict(self.Q,new_q,s0,a0) # 更新s0,a0的Q值
            s0,a0 = s1,a1
            timestep_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return timestep_in_episode,total_reward

class SarsalambdaAgent(Agent):
    def __init__(self,env:Env,capacity:int=20000):
        super(SarsalambdaAgent,self).__init__(env,capacity)
        self.Q = {}

    def policy(self,A,s=None,Q=None,epsilon=None):
        return epsilon_greedy_policy(A,s,Q,epsilon)

    def learning_method(self,lambda_=0.9,gamma=0.9,alpha=0.1,epsilon=0.2,
            display=False,current_episode=None):
        self.state = self.env.reset()
        s0 = self.state
        if current_episode > 200:
            self.env.render()
        a0 = self.perform_policy(s0,self.Q,epsilon)

        timestep_in_episode=0
        total_reward = 0
        is_done = False
        E = {} # 资格迹
        while not is_done:
            s1,r1,is_done,info,total_reward = self.act(a0)
            if current_episode > 200 and display==True:
                self.env.render()
            a1 = self.perform_policy(s1,self.Q,epsilon)

            q = get_dict(self.Q,s0,a0)
            q_prime = get_dict(self.Q,s1,a1)
            delta = r1 + gamma*q_prime - q

            e = get_dict(E,s0,a0)
            e += 1
            set_dict(E,e,s0,a0)

            for s in self.S:
                for a in self.A:
                    e_value =  get_dict(E,s,a)
                    old_q = get_dict(self.Q,s,a)
                    new_q = old_q + alpha*delta*e_value
                    new_e = gamma*lambda_*e_value
                    set_dict(self.Q,new_q,s,a)
                    set_dict(E,new_e,s,a)

            s0,a0 = s1,a1
            timestep_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return timestep_in_episode,total_reward

if __name__ == "__main__":
    env = CliffWalk2()
    # agent = SarsaAgent(env,capacity=10000)
    agent2 = SarsalambdaAgent(env,capacity=10000)
    #total_times,episode_rewards,num_episodes
    # data = agent.learning(gamma=1.0,
    #                       epsilon=1,
    #                       decay_epsilon=True,
    #                       alpha=0.5,
    #                       max_episode_num=800,
    #                       display=True)

    data2 = agent2.learning(lambda_=0.8,gamma=1.0,epsilon=0.2,
                  decay_epsilon=True,alpha=0.5,max_episode_num=800,display=True)
    learning_curve(data2,2,1,x_name="Episodes",y_name="Rewards",title="SARSA Agent")
