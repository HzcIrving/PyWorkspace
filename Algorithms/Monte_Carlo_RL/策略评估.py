#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from 正式对局 import *
from 工具函数 import *


def str_key(*args):
    '''将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value

def get_dict(target_dict, *args):
    return target_dict.get(str_key(*args), 0)

# print(arena.episodes)

# 对局生成的数据均保存在对象episodes中
# 使用这些数据对玩家的策略进行评估
# gamma = 1 衰减因子为1
def policy_evaluate(episodes,V,Ns):
    for episode,r in episodes:
        for s,a in episode:
            ns = get_dict(Ns,s) # 状态被防卫的次数节点
            v = get_dict(V,s) # 状态价值
            set_dict(Ns,ns+1,s)
            set_dict(V,v+(r-v)/(ns+1),s) # Monte-Carlo Evaluation
# 绘制价值函数
def draw_value(value_dict,useable_ace=True,is_q_dict=False,A=None):
    fig = plt.figure()
    ax = Axes3D(fig)

    # 定义x,y
    x = np.arange(1,11,1) # 庄家第一张牌
    y = np.arange(12,22,1) # 玩家总分数

    # 生成网格
    X,Y = np.meshgrid(x,y)
    # 从V字典检索Z的高度
    row,col = X.shape
    Z = np.zeros((row,col))
    if is_q_dict:
        n = len(A)
    for i in range(row):
        for j in range(col):
            state_name = str(X[i,j])+"_"+str(Y[i,j])+"_"+str(useable_ace)
            if not is_q_dict:
                Z[i,j] = get_dict(value_dict,state_name) #查询对应的state value值
            else:
                assert (A is not None)
                for a in A:
                    new_state_name = state_name+"_"+str(a)
                    q = get_dict(value_dict,new_state_name)
                    if q >= Z[i,j]:
                        Z[i,j] = q  # Q值，MC中不需要

    # 绘制3D曲面
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,color='lightgray')
    # plt.show()


if __name__ == '__main__':
    V = {} # 状态价值字典
    Ns = {} # 状态被访问的次数节点
    # print(arena.episodes)
    policy_evaluate(arena.episodes,V,Ns) # 学习评估V值

    draw_value(V,useable_ace=True,A=A) # 绘制有可用A的时候的状态价值图
    draw_value(V,useable_ace=False,A=A) # 绘制无可用A的时候的状态价值图
    plt.show()
