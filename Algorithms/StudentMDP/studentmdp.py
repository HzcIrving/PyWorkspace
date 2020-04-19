#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
DavidSilver强化学习第一课
加深对：
    Markov Reward Process;
    Markov Decision Process;
    Return & Value；
    Bellman期望方程；
    Bellman最优方程；
理解

1.收获和价值计算
S: 7个， C1,C2,C3,FB,Pub,Pass,Sleep
P_ss': 状态转换概率--7x7矩阵
R: 奖励函数7个标量--即使奖励值

"""
import numpy as np
from utils import *

num_states = 7

# {
#  "0":"C1","1":"C2","2":"C3","3":"Pass","4":"Pub","5":"FB",
#  "6":"Sleep"
# }
i_to_n = {} # 索引到状态名的字典
i_to_n["0"]="C1"
i_to_n["1"]="C2"
i_to_n["2"]="C3"
i_to_n["3"]="Pass"
i_to_n["4"]="Pub"
i_to_n["5"]="FB"
i_to_n["6"]="Sleep"

n_to_i = {} #状态名称到索引的字典
for i,name in zip(i_to_n.keys(),i_to_n.values()):
    n_to_i[name]=int(i)

print(i_to_n)
print(n_to_i)

# C1   C2   C3  Pass Pub   FB   Sleep
Pss = [ # 状态转移概率矩阵
[ 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0 ],
[ 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2 ],
[ 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0 ],
[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
[ 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0 ],
[ 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
]

Pss = np.array(Pss)

# REWARDS：
rewards = [-2,-2,-2,10,1,-1,0]
gamma = 0.5 # 衰减因子

# 例1：计算回报return值 Gt
def compute_return(start_index=0,chain=None,gamma=0.5)->float:
    """
    计算一个马尔科夫奖励过程中某状态的Gt
    Gt = R[t+1]+gamma*R[t+2]+gamma^2*R[t+3]...
    从某一状态St开始，直到终止状态时，All Rewards的衰减值
    :param start_index: 要计算的状态在链中的位置
    :param chain: 要计算的Markov过程
    :param gamma: 衰减系数
    :return:
        retrn 收获值/回报值
    """
    retrn,power,gamma = 0.0,0,gamma
    for i in range(start_index,len(chain)):
        retrn += np.power(gamma,power)*rewards[n_to_i[chain[i]]]
        power += 1
    return retrn

# 例子中几个Markov Chain定义如下：
chains =[
["C1", "C2", "C3", "Pass", "Sleep"],
["C1", "FB", "FB", "C1", "C2", "Sleep"],
["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB",\
"FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

print("ReturnChain1:",compute_return(0,chains[0],gamma=0.5))
print("ReturnChain2:",compute_return(0,chains[1],gamma=0.5))


# 例2：使用矩阵运算求解各个状态的价值
def compute_value(Pss,rewards,gamma=1):
    """
    计算状态价值value
    V(s)=R[t+1] + gamma * sum(Pss'*V(s'))
    :param Pss: 状态转移概率矩阵 7x7
    :param rewards: 即时奖励list
    :param gamma: 衰减
    :return:  values各个状态的价值
    """
    rewards = np.array(rewards).reshape(-1,1) # 转为列向量
    # V = R + gamma * P*V
    # V = (1-gamma*P)^(-1) * R
    values = np.dot(np.linalg.inv(np.eye(7,7)-gamma*Pss),rewards)
    return values

# 按照例题
values = compute_value(Pss,rewards,gamma=0.9999) # gamma=1会出现奇异问题
print(values)

# 例3: Bellman方程
# 此时：状态是5个 "浏览手机中"，"第一节课"，"第二节课"，"第三节课"，"休息中"
#      行为是5个 "浏览手机"，"学习"，"放下手机"，"泡吧"，"退出学习"

S = ["浏览手机中","第一节课","第二节课","第三节课","休息中"]
A = ["浏览手机","学习","放下手机","泡吧","退出学习"]
R = {} # 奖励 Rsa 字典
P = {} # 状态转移概率 Pss'^a 字典
gamma = 1.0 # 衰减因子
Pi = {}

# 根据student Markov Decision Process示例的数据设置状态转移概率和奖励；
# 默认概率为1，在DavidSilver的例子中，大部分时刻，选择action后，下一状态确定了
# 除了pub状态，有c1,c2,c3三种选择
set_prob(P,S[0],A[0],S[0]) # 浏览手机中->浏览手机->浏览手机中
set_prob(P,S[0],A[2],S[1]) # 浏览手机中->关闭手机->第一节课
set_prob(P,S[1],A[0],S[0]) # C1->浏览手机->浏览手机中
set_prob(P, S[1], A[1], S[2]) # ...
set_prob(P, S[2], A[1], S[3]) # ...
set_prob(P, S[3], A[1], S[4])
set_prob(P, S[3], A[3], S[1], p = 0.2) # 选择pub的action后 0.2几率选择C1
set_prob(P, S[3], A[3], S[2], p = 0.4)
set_prob(P, S[3], A[3], S[3], p = 0.4)

set_reward(R,S[0],A[0],-1) # 浏览手机中选择浏览手机 R = -1
set_reward(R,S[0],A[2],0) # 浏览手机中选择关闭手机 R = 0
set_reward(R, S[1], A[0], -1)
set_reward(R, S[1], A[1], -2)
set_reward(R, S[2], A[1], -2)
set_reward(R, S[2], A[4], 0)
set_reward(R, S[3], A[1], 10) # Pass and Sleep
set_reward(R, S[3], A[3], +1) # 泡吧奖励

set_pi(Pi,S[0],A[0],0.5) # 浏览手机中时选择浏览手机概率为0.5
set_pi(Pi, S[0], A[2], 0.5)
set_pi(Pi, S[1], A[0], 0.5)
set_pi(Pi, S[1], A[1], 0.5)
set_pi(Pi, S[2], A[1], 0.5)
set_pi(Pi, S[2], A[4], 0.5)
set_pi(Pi, S[3], A[1], 0.5)
set_pi(Pi, S[3], A[3], 0.5)


MDP = (S,A,R,P,gamma)

print("-----状态转移概率字典(矩阵)信息-----")
display_dict(P)
# -----状态转移概率字典(矩阵)信息-----
# 浏览手机中_浏览手机_浏览手机中:1.00
# 浏览手机中_放下手机_第一节课:1.00
# 第一节课_浏览手机_浏览手机中:1.00
# 第一节课_学习_第二节课:1.00
# 第二节课_学习_第三节课:1.00
# 第三节课_学习_休息中:1.00
# 第三节课_泡吧_第一节课:0.20
# 第三节课_泡吧_第二节课:0.40
# 第三节课_泡吧_第三节课:0.40

print("-----奖励字典(函数)信息-----")
display_dict(R)
# -----奖励字典(函数)信息-----
# 浏览手机中_浏览手机:-1.00
# 浏览手机中_放下手机:0.00
# 第一节课_浏览手机:-1.00
# 第一节课_学习:-2.00
# 第二节课_学习:-2.00
# 第二节课_退出学习:0.00
# 第三节课_学习:10.00
# 第三节课_泡吧:1.00

print("-----策略选择概率字典信息------")
display_dict(Pi)
# -----策略选择概率字典信息------
# 浏览手机中_浏览手机:0.50
# 浏览手机中_放下手机:0.50
# 第一节课_浏览手机:0.50
# 第一节课_学习:0.50
# 第二节课_学习:0.50
# 第二节课_退出学习:0.50
# 第三节课_学习:0.50
# 第三节课_泡吧:0.50

"""
有了上述信息，编写代码静计算在给定MDP和状态价值函数V的条件下，
如何计算某一状态s时，某行为a的价值q(s,a)
在给定策略Pi下，计算某一状态s的价值
"""
def compute_q(MDP,V,s,a):
    """
    给定MDP，价值函数V,计算状态行为对s,a的价值q(s,a)
    :param MDP: (S,A,R,P,gamma)
    :param V: Value Function
    :param s: state
    :param a: action
    :return: q_sa 行为状态值函数
    """
    S,A,R,P,gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P,s,a,s_prime)*get_value(V,s_prime)
    q_sa = get_reward(R,s,a) + gamma*q_sa # 公式
    return q_sa

def compute_v(MDP,V,Pi,s):
    """
    给定策略Pi下，计算某一状态值
    :param MDP: ...
    :param V: ...
    :param Pi: ...
    :param s: ...
    :return: v_s
    """
    S,A,R,P,gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi,s,a)*compute_q(MDP,V,s,a)
    return v_s

"""
上面两个function，让我们明确了基于某一策略下，各状态以及
各状态行为对的价值。
下面，需要给出在均一随机策略下，该学生Markov决策过程的最终
状态函数。
"""

# 根据当前策略，使用回溯法，来更新状态价值
def update_V(MDP,V,Pi):
    # 给定一个MDP和一个策略，更新该策略下价值函数V
    S,_,_,_,_ = MDP
    V_prime = V.copy()
    for s in S:
        # set_value(V_prime, s, V_S(MDP, V_prime, Pi, s))
        V_prime[str_key(s)] = compute_v(MDP,V_prime,Pi,s)
    return V_prime

# 策略评估，得到该策略下最终的状态价值
def policy_evaluate(MDP,V,Pi,n):
    """使用n此迭代计算来评估一个MDP在给定策略Pi下的状态价值"""
    for i in range(n):
        V = update_V(MDP,V,Pi)
        # display_dict(V)
    return V

V = {}
V = policy_evaluate(MDP,V,Pi,100)
display_dict(V)

v = compute_v(MDP,V,Pi,"第三节课")
print("第三节课在当前策略下的最终价值:{:.2f}".format(v))
# 浏览手机中:-2.31
# 第一节课:-1.31
# 第二节课:2.69
# 第三节课:7.38
# 休息中:0.00

"""
上面的结果与pdf中的示例结果相同；
不同策略得到状态的最终价值不同，最优策略下，最优状态价值计算遵循
课本2.22公式
"""
def compute_v_from_max_q(MDP,V,s):
    """
    根据一个状态下的所有可能的行为价值中最大一个来确定当前状态价值
    Bellman最优公式
    """
    S,A,R,P,gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP,V,s,a)
        if qsa >= v_s:
            v_s = qsa
    return v_s

def update_V_without_pi(MDP,V):
    """在不依赖策略的情况下，直接通过后续状态价值更新状态价值"""
    S,_,_,_,_ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v_from_max_q(MDP,V_prime,s)
    return V_prime

# 价值迭代
def value_iterate(MDP,V,n):
    for i in range(n):
        V = update_V_without_pi(MDP,V)
    return V

V = {}
V_star = value_iterate(MDP,V,4) # 4次
display_dict(V_star)
# 浏览手机中:6.00
# 第一节课:6.00
# 第二节课:8.00
# 第三节课:10.00
# 休息中:0.00
# 验证了课本中的例子

# 有了最优状态价值，可以计算最优行为价值
s,a = "第三节课","泡吧"
q = compute_q(MDP,V_star,s,a)
print("在状态{}选择行为{}的最优价值为:{:.2f}".format(s,a,q)) # 9.4 符合
