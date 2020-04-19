#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""

状态序列管理(个体与环境进行交互会产生一个或者多个甚至大量的状态序列，
如何管理好这些状态序列是编程实践环境一个重要任务)
1.状态序列是时间序列
--- 每个time step,一个complete状态转换:(St,At,Rt+1,St+1,done)
2.每个相邻状态转换构成一个状态序列
--- 多个完整的状态序列形成了个体的整体experience/memory
3. MC学习方法
--- 需要完整的episode状态序列；
4. TD学习
--- 最小的学习单位是一个状态转换;
5. 许多TD选择不连续状态转换来学习
--- 降低TD学习在一个序列中的偏差，这种situation倒不一定非要按照时间次序
--- 形式进行管理；

class: Transition
class: Episode
class: Experience

"""

from random import random,choice
import gym
from gym import Env
import numpy as np
from collections import namedtuple
from typing import List
import random
from tqdm import tqdm

class State(object):
    def __init__(self,s0,a0,reward:float,is_done:bool,s1):
        self.data = [s0,a0,reward,is_done,s1]

# 状态转换
class Transition(object):
    # 状态转换
    def __init__(self,s0,a0,reward:float,is_done:bool,s1):
        self.data = [s0,a0,reward,is_done,s1]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return "s:{0:<3} a:{1:<3} r:{2:<4} is_end:{3:<5} s1:{4:<3}".\
            format(self.data[0],self.data[1],self.data[2],
                   self.data[3],self.data[4])

    @property
    def s0(self): return self.data[0]

    @property
    def a0(self): return self.data[1]

    @property
    def reward(self): return self.data[2]

    @property
    def is_done(self): return self.data[3]

    @property
    def s1(self): return self.data[4]

# 状态序列
class Episode(object):
    # 完整状态序列
    def __init__(self,e_id:int = 0) -> None : #默认返回None
        self.total_reward = 0 # 总奖励
        self.trans_list = [] # 状态转移列表
        self.name = str(e_id) # 状态序列名字属性

    def push(self,trans:Transition) -> float:
        self.trans_list.append(trans) # 添加状态转移
        self.total_reward += trans.reward
        return self.total_reward

    @property
    def len(self):
        # 属性，状态列表长度
        return len(self.trans_list)

    def __str__(self):
        # <4指间隙为4个字符
        return "episode {0:<4} {1:>4} steps,total reward:{2:<8.2f}"\
            .format(self.name,self.len,self.total_reward)

    def print_detail(self):
        # 打印状态序列信息
        print("detail of ({0}):".format(self))
        for i,trans in enumerate(self.trans_list):
            print("step{0:<4}".format(i),end="")
            print(trans)

    def pop(self) -> Transition:
        # 删除状态序列中首个状态转移状态
        if self.len>1:
            trans = self.trans_list.pop()
            self.total_reward -= trans.reward
            return trans
        else:
            return None

    def is_complete(self) -> bool:
        # 检查状态序列是否结束
        if self.len == 0:
            return False
        return self.trans_list[self.len-1].is_done

    def sample(self,batch_size=1):
        # 随机产生一个状态转移
        return random.sample(self.trans_list,k=batch_size)

    def __len__(self) ->int :
        return self.len

# 经验池 Memory/Experience
class Experience(object):
    """记录智能体整个状态序列列表"""
    def __init__(self,capacity:int=20000):
        """默认内存20000个"""
        self.capacity = capacity  # 容量，指的是trans总数量
        self.episodes = [] # episode列表
        self.next_id = 0  # 下一个episode的id
        self.total_trans = 0  # 总的状态转换数量

    def __str__(self):
        return "exp info:{0:5} episode, memory usage{1}/{2}".\
            format(self.len,self.total_trans,self.capacity)

    def __len__(self):
        return self.len

    @property
    def len(self):
        # 包含的状态序列总数
        return len(self.episodes)

    def _remove(self,index=0):
        if index > self.len-1:
            raise(Exception("Invalid Index Input"))
        if self.len>0:
            episode = self.episodes[index]
            self.episodes.remove(episode) # 移除该状态序列
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def _remove_first(self):
        self._remove(index=0)

    def push(self,trans):
        """压入一个状态转换"""
        if self.capacity <=0:
            return
        while self.total_trans >= self.capacity: # 如果当前总数大于容量
            episode = self._remove(0) # 移除第一个
        cur_episode = None
        if self.len == 0 or self.episodes[self.len-1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1 # 开始添加下个episode
            self.episodes.append(cur_episode) # 一个状态序列结束才能添加
        else:
            cur_episode = self.episodes[self.len-1]
        self.total_trans+=1
        return cur_episode.push(trans)

    def sample(self,batch_size=1):
        """随机采样：随机获取一定数量的状态转化对象Transition"""
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random()) * self.len # 定为episode
            sample_trans += self.episodes[index].sample() #从状态序列中采样transition
        return sample_trans

    def sample_episode(self,episode_num = 1):
        """随机采样整个状态序列"""
        return random.sample(self.episodes,k=episode_num)

    @property
    def last_episode(self):
        """输出最后一个状态序列"""
        if self.len>0:
            return self.episodes[self.len-1]
        return None

#
#
#
# if __name__ == "__main__":
#     trans1 = Transition(0,1,1,False,5)
#     trans2 = Transition(1,2,3,True,6)
#     trans3 = Transition(0,2,4,False,0)
#     trans4 = Transition(1,5,1,True,0)
#
#     exp = Experience()
#     exp.push(trans1)
#     exp.push(trans2)
#     exp.push(trans3)
#     exp.push(trans4)
#
#     print(exp.len)
#     # exp.sample(1).print_details()
#     # print(exp.sample_episode(1))
#     exp.sample_episode(1)
#     print(exp.last_episode)
#     # print(trans1.s0)
#     # print(trans1.is_done)
#     # print(trans1.__str__())
#     # ep1 = Episode()
#     # ep1.push(trans1)
#     # print(ep1.len)
#     # ep1.print_detail()









