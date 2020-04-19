#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
基于MC学习求二十一点游戏最优策略

GLIE:
Greedy in the Limit with Infinite Exploration
1. 所有的(s,a)都会被无限次探索;
2. 随着采样趋于∞,策略收敛至一个贪婪策略;

P-code
Loop:
{
    1.依据π，确定第k个episode:{S1,A1,R2,...,ST}
    2.对每个st,at in St,At:
        2.1 N(st,at) ← N(st,at)+1
        2.2 Q(st,at) ← Q(st,at)+1/N(st,at)(Gt-Q(st,at))
    3.更新epsilon:
        3.1 epsilon ← 1/k
    4.π~epsilon-greedy
}
"""

from 决胜21点 import Player,Dealer
from 赌桌环境 import Arena
from 工具函数 import str_key,set_dict,get_dict
from 策略 import epsilon_greedy_policy,epsilon_greedy_pi
import math

# 目前Player不具备策略评估和更新策略的能力
class MC_Player(Player):
    """给Player赋予策略控制能力"""

    def __init__(self,name="",A=None,display=False):
        super(MC_Player,self).__init__(name,A,display)
        self.Q = {} # 某一状态行为对的价值 字典， 在策略迭代时使用
        self.Nsa = {} # s,a状态行为对计数
        self.total_learning_times = 0
        self.policy = self.epsilon_greedy_policy
        self.learning_method = self.learn_Q

    def epsilon_greedy_policy(self,dealer,epsilon=None):
        """epsilon greedy策略"""
        player_points,_ = self.get_points()
        if player_points >= 21:
            return self.A[1]
        if player_points < 12:
            return self.A[0] # 叫
        else:
            A,Q = self.A,self.Q
            # 12~21之间的选择
            s = self.get_state_name(dealer) # 获取庄家的第一张牌面
            if epsilon is None :
                epsilon = 1.0/(1+4*math.log10(1+player.total_learning_times))
            return epsilon_greedy_policy(A,s,Q,epsilon) # 根据epsilon-greedy选择策略

    def learn_Q(self,episode,r):
        """从一个episode来学习Q值"""
        for s,a in episode:
            nsa = get_dict(self.Nsa,s,a)
            set_dict(self.Nsa,nsa+1,s,a)
            q = get_dict(self.Q,s,a)
            set_dict(self.Q,q+(r-q)/(nsa+1),s,a) # loop 更新s,a的Q
        self.total_learning_times+=1

    def reset_memory(self):
        """忘记既往学习经历"""
        self.Q.clear()
        self.Nsa.clear()
        self.total_learning_times = 0


if __name__ == '__main__':
    A = ['叫牌','停止叫牌']
    display=False
    player = MC_Player(A=A,display=display) # 此时玩家是MC_Player
    dealer = Dealer(A=A,display=display)
    arena = Arena(A=A,display=display)
    arena.play_games(dealer=dealer,player=player,num=200000,show_statistic=True)
