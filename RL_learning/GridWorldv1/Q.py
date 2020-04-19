#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""Q-Learning"""

from random import  random,choice
from gym import Env
from Toolutils import  *
from GridWorldv1 import  *
from DifferentGridEnv import *
import gym
from Agent import  Agent

class QAgent(Agent):
    def __init__(self,env:Env,capacity:int=20000):
        super(QAgent,self).__init__(env,capacity)
        self.Q = {}

    def policy(self,A,s=None,Q=None,epsilon=None):
        return epsilon_greedy_policy(A,s,Q,epsilon)

    def learning_method(self,lambda_=0.9,gamma=0.9,alpha=0.5,epsilon=0.2,
            display=False,current_episode=None):
        self.state = self.env.reset()
        s0 = self.state
        if current_episode>50 and display==True:
            self.env.render()

        timestep_in_episode,total_reward = 0,0
        is_done = False
        while not is_done:
            # self.policy = epsilon_greedy_policy
            a0 = self.perform_policy(s0,self.Q,epsilon) # u
            s1,r1,is_done,info,total_reward=self.act(a0)
            if current_episode>50 and display==True:
                self.env.render()
            # self.policy = greedy_policy
            a1 = greedy_policy(self.A,s1,self.Q) # pi offpolicy
            old_q = get_dict(self.Q,s0,a0)
            q_prime = get_dict(self.Q,s1,a1)
            td_target = r1+gamma*q_prime
            new_q = old_q + alpha*(td_target-old_q)
            set_dict(self.Q,new_q,s0,a0)
            s0=s1
            timestep_in_episode+=1

        if display:
            print(self.experience.last_episode)
        return timestep_in_episode,total_reward

if __name__ == "__main__":
    env = CliffWalk2()
    agent = QAgent(env)
    #
    data = agent.learning(gamma=1.0,
                          epsilon=0.1,
                          decay_epsilon=True,
                          alpha=0.5,
                          max_episode_num=800,
                          display=False)

    # agent.learning_method(epsilon=0.01,display=True)
    learning_curve(data,x_index=2,y1_index=1,title="Q_Learning_curve",x_name="Episodes",
                   y_name="rewards")


