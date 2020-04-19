#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

def str_key(*args):
    """
    将参数用"_"连接作为字典的key
    参数本身可能是tuple或者List --- ((a,b,c),d)
    :param args:
    :return:
    """
    new_arg = []
    for arg in args:
        if type(arg) in [tuple,list]:
            # 检测arg类型
            # 元组, i.g. (a,b,c) new_arg = abc
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict,value,*args):
    target_dict[str_key(*args)] = value
    # return target_dict

def set_prob(P,s,a,s1,p=1.0): #设置字典概率
    # default 概率为1
    dict = set_dict(P,p,s,a,s1)
    # return dict

def set_reward(R,s,a,r): # 设置字典奖励
    set_dict(R,r,s,a)

def set_pi(Pi,s,a,p=0.5):
    """
    设置策略字典
    :param Pi: 策略
    :param s:  状态值
    :param a:  选择某一action的值
    :param p:  选择a的概率p
    """
    set_dict(Pi,p,s,a)

def set_value(V,s,v): # 设置价值字典
    set_dict(V,v,s)

def get_prob(P,s,a,s1): # 获取概率值
    return P.get(str_key(s,a,s1),0)

def get_reward(R,s,a): # 获取即时奖励值
    return R.get(str_key(s,a),0)

def get_value(V,s): # 获取价值值
    return V.get(str_key(s),0)

def get_pi(Pi,s,a): # 获取策略值
    return Pi.get(str_key(s,a),0)

def display_dict(target_dict): # 显示字典内容
    for key in target_dict.keys():
        print("{}:{:.2f}".format(key,target_dict[key]))
    print("")

if __name__ == "__main__":
    # S = ["a", "b", "c", "d", "e"]
    S = ["浏览手机中", "第一节课", "第二节课", "第三节课", "休息中"]
    A = ["浏览手机", "学习", "放下手机", "泡吧", "退出学习"]
    R = {}  # 奖励 Rsa 字典
    P = {}  # 状态转移概率 Pss'^a 字典
    # print(str_key(S))
    # print(set_prob(P,S[0],A[0],S[0])) # 浏览手机中->浏览手机->浏览手机中
    # print(set_prob(P,S[0],A[2],S[1]))  # 浏览手机中->关闭手机->第一节课