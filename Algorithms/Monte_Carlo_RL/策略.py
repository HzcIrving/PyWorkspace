#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from 工具函数 import get_dict,set_dict
import random

def greedy_pi(A,s,Q,a):
    """
    依据greedy思想，计算在action space A中，状态s下，
    a行为被贪婪选中的几率;
    考虑多个行为价值相同的情况
    """
    max_q,a_max_q = -float('inf'),[]
    for a_opt in A:
        q = get_dict(Q,s,a_opt) # 获得Q值
        if q > max_q:
            max_q = q # 更新最大Q值
            a_max_q = [a_opt] # 记录当前最大Q值的action
        elif q == max_q:
            a_max_q.append(a_opt) #若存在相同的最大Q值，同样记录下action
    n = len(a_max_q)
    if n == 0:
        return 0.0 # 无策略
    return 1.0/n if a in a_max_q else 0.0
    # 对相同的最大Q值对应的action等概率选取

def greedy_policy(A,s,Q):
    """
    在给定一个状态下，从action space A中选择一个行为a,使得
    Q(s,a) = max(Q(s,))
    """
    max_q,a_max_q = -float('inf'),[]
    for a_opt in A:
        q = get_dict(Q,s,a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)
    # 从可以获得最大Q对应的action随机选取

def epsilon_greedy_pi(A,s,Q,a,epsilon=0.1):
    """epsilon greedy策略"""
    m = len(A)
    greedy_p = greedy_pi(A,s,Q,a)
    if greedy_p == 0:
        return epsilon/m # 非greedy的随机策略 1/m*epsilon  概率
    n = int(1.0/greedy_p)
    return (1-epsilon)*greedy_p + epsilon/m  # greedy的策略的概率

def epsilon_greedy_policy(A,s,Q,epsilon,show_random_num=False):
    """epsilon greedy策略"""
    pis = []
    m = len(A)
    for i in range(m):
        pis.append(epsilon_greedy_pi(A,s,Q,A[i],epsilon)) # 返回选择a*的概率
    rand_value = random.random() # 产生一个0,1的随机数

    for i in range(m):
        if show_random_num:
            print("随机数:{:.2f},拟减去概率{}".format(rand_value,pis[i]))
        rand_value -= pis[i]
        if rand_value < 0:  # 若rand_value小于epsilon 则随机选择一个
            return A[i]
