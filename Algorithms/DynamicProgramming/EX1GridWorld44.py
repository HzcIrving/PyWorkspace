#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
叶强课件练习
c03.pdf
4x4 grid world
↖0和↘15是final state,agent到达任意一个R=0,其余R=1
"""
import numpy as np
import math
# from utils import *

"""
先对4x4小型方格世界进行建模
该环境动力学明确s
"""
S = [i for i in range(16)] # 状态空间
A = ["n","e","s","w"] # 上、下、左、右
# P,R 由dynamics动态生成

# 选择n state-4
# 选择e state+1
# 选择s state+4
# 选择w state-1
ds_actions = {"n":-4,"e":1,"s":4,"w":-1} # 行为对状态的改变

def dynamics(s,a):
    """
    环境动力学，grid world的环境动力学特征
    :param s: 当前状态 int 0~15
    :param a: 行为str，北、东、南、西
    :return: tuple(s',reward,is_end)
        s' 后续状态, reward 即时奖励, is_end 终点标志位
    """
    s_prime = s
    # 控制边界问题
    if (s%4==0 and a =="w") or (s<4 and a =="n") \
        or ((s+1)%4==0 and a=="e") or (s>11 and a=='s')\
        or s in [0,15]: # 排除边界情况与直接在终点的情况
        pass
    else:
        ds = ds_actions[a] # 对应的state_value增量
        s_prime = s+ds
    reward = 0 if s in [0,15] else -1
    is_end = True if s in [0,15] else False

    return s_prime,reward,is_end

def P(s,a,sl):
    """状态转移概率函数"""
    s_prime,_,_ = dynamics(s,a)
    return sl==s_prime

def R(s,a):
    """奖励函数"""
    _,r,_ = dynamics(s,a)
    return r

# 辅助函数
def get_prob(P,s,a,s1):
    """注意，这里的P是函数"""
    # 获取状态转移概率
    return P(s,a,s1)

def get_reward(R,s,a):
    """注意，这里的R是函数"""
    # 获取奖励值
    return R(s,a)

def set_value(V,s,v):
    """设置价值字典"""
    V[s] = v

def get_value(V,s):#获取状态价值
    return V[s]

def display_V(V): #显示状态价值
    for i in range(16):
        print('{0:>6.2f}'.format(V[i]),end="")
        if (i+1)%4 == 0:
            print("") # 4x4环境
    print()

# 设置衰减
gamma = 1.00
MDP = S,A,R,P,gamma # RP都是函数

"""
构建策略函数
均一策略：4个方向探索概率相同;
贪婪策略：pi'(s) = argmax(a∈A)q_pi(s,a)
"""
def uniform_random_pi(MDP=None,V=None,s=None,a=None):
    """均一策略"""
    _,A,_,_,_ = MDP
    n = len(A)
    return 0 if n==0 else 1.0/n

def greedy_pi(MDP,V,s,a):
    """贪婪策略"""
    S,A,P,R,gamma=MDP
    # 初始化
    max_v,a_max_v = -float('inf'),[]
    for a_opt in A:
        # 统计后续状态的最大价值以及到达该状态的行为(可能不止1个)
        s_prime,reward,_ = dynamics(s,a_opt)
        v_s_prime = get_value(V,s_prime) # 获得V'
        if v_s_prime > max_v:
            max_v = v_s_prime # 改变max_v
            a_max_v = [a_opt] # 记录下此时获得max_v时的A
        elif (v_s_prime == max_v):
            a_max_v.append(a_opt) # 记录下此时的a_opt
    n = len(a_max_v)
    if n == 0: return 0.0
    # n = 1 就选择此时的a
    # n != 1 则对这些a等概率选取
    return 1.0/n if a in a_max_v else 0.0

def get_pi(Pi,s,a,MDP=None,V=None):
    """策略选择 Pi是输入的策略选择函数，可以是均一策略，也可以是贪婪策略"""
    return Pi(MDP,V,s,a)

#----TASK1 prediction预测 迭代法策略评估----
def compute_q(MDP,V,s,a):
    """根据给定MDP，价值函数V,计算状态行为对s,a的价值qsa"""
    S,A,R,P,gamma = MDP
    # 注意，此时R,P是函数
    q_sa = 0
    # get_prob和get_reward均调用了P和R函数
    for s_prime in S:
        q_sa += get_prob(P,s,a,s_prime)*get_value(V,s_prime)
    q_sa = get_reward(R,s,a) + gamma*q_sa
    return q_sa

def compute_v(MDP,V,Pi,s):
    """根据给定MDP，依据某个策略Pi和当前状态价值函数V计算某状态S的价值"""
    S,A,R,P,gamma = MDP
    v_s = 0
    for a in A:
        # get_pi获取当前策略
        # 这个策略依据均一策略函数或者贪婪策略函数
        # uniform_random_pi
        # 或者 greedy_pi
        v_s += get_pi(Pi,s,a,MDP,V)*compute_q(MDP,V,s,a)
    return v_s

def update_V(MDP,V,Pi):
    """给定一个MDP和一个策略，更新该策略下的价值函数V"""
    S,_,_,_,_= MDP
    V_prime = V.copy() # 初始化
    for s in S:
        set_value(V_prime,s,compute_v(MDP,V_prime,Pi,s))
    return V_prime

def policy_evaluate(MDP,V,Pi,n):
    """策略估计，使用n次迭代计算，评估一个MDP在给定策略Pi下的状态价值，
        初始时价值为V"""
    for i in range(n):
        # print("====第{}次迭代====".format(i+1))
        V = update_V(MDP,V,Pi)
        # display_V(V)
    return V

V = [0 for _ in range(16)] # 状态价值
V_pi = policy_evaluate(MDP,V,uniform_random_pi,100)
display_V(V_pi) #  以均一策略迭代100次的结果
V = [0 for _ in range(16)] # 状态价值
V_pi = policy_evaluate(MDP,V,greedy_pi,100)
display_V(V_pi)

#----TASK2 control_1控制 策略迭代----
def policy_iterate(MDP,V,Pi,n,m):
    """
    策略迭代
    :param MDP: Markov Decision Process
    :param V: Value
    :param Pi: Policy
    :param n: 逐次迭代
    :param m: 策略迭代次数
    :return: V最优策略
    """
    for i in range(m):
        V = policy_evaluate(MDP,V,Pi,n)
        Pi = greedy_pi # 第一次迭代产生新的价值函数后使用随机贪婪策略
    return V

V = [0 for _ in range(16)]
V_pi = policy_iterate(MDP,V,greedy_pi,1,100)
display_V(V_pi)

#----TASK3 control_2控制 价值迭代----
#价值迭代得到最优状态价值
def compute_v_from_max_q(MDP,V,s):
    """根据一个状态的后续所有可能的行为价值中的最大一个来确定当前状态价值"""
    S,A,R,P,gamma = MDP
    v_s = -float("inf")
    for a in A:
        qsa = compute_q(MDP,V,s,a)
        if qsa>=v_s:
            v_s = qsa # 最大原则 Bellman最优
    return v_s

def update_V_without_pi(MDP,V):
    """不依赖策略的情况下，直接通过后续状态价值来更新状态价值"""
    S,_,_,_,_ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime,s,compute_v_from_max_q(MDP,V_prime,s)) # V_prime更新
    return V_prime

def value_iterate(MDP,V,n):
    """价值迭代"""
    for i in range(n):
        V = update_V_without_pi(MDP,V)
        display_V(V)
    return V

V = [0 for _ in range(16)] #重置状态价值
display_V(V)
V_star = value_iterate(MDP,V,5) # 迭代5次
display_V(V_star) # 5次就迭代到最优~






