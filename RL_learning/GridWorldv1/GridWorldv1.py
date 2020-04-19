#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
基于gym库的Grid World
具备记忆功能

gym:提供了一整套编程接口和丰富的强化学习环境，同时还具备可视化功能；
    库核心文件： core.py
    基本类： Env 和 Space
        其中 Space中衍生： Discrete和Box类，前者一维离散空间，后者多维连续空间
        即可描述action space也可描述state space

4x4 格子世界：
一共16个状态，每一个状态只需要一个数子描述---Discrete(16) state space
每次上下左右四个动作---Discrete(4) action space

Env结构:
--- step() 环境动力学 确定个体的下一个状态，奖励信息，个体是否到达终点等；
--- reset() 重置环境
--- seed() 随机数种子
--- render() 简单可视化工作，如果需要将个体与环境的交互以动画形式展现，需要重写
             简单的UI设计可以通过gym包装好的了pyglet来写

具体包括：
--- 带有风干扰；
--- 悬崖行走；
--- 随机行走；
...

"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# 每个Grid的属性
class Grid(object):
    """定义<格子世界>中每个格子的属性；"""
    def __init__(self,x:int=None,
                      y:int=None,
                      type:int = 0,
                      reward:float = 0.0,
                      value:float = 0.0): # value属性备用
        self.x = x # 坐标x
        self.y = y # 坐标y
        self.type = type  # 类别值(0:空; 1:障碍或者边界)
        self.reward = reward # 该格子的即时奖励
        self.value = value # 该格子的价值
        self.name = None # 该格子的名称
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x,self.y)

    def _str(self):
        return "Name:{4},x:{0},y:{1},Type:{2},Value:<<{3}>>".format(self.x,\
                                self.y,self.type,self.reward,self.value,self.name)

# 格子矩阵，通过不同的设置，模拟不同的格子世界环境
# 各种格子世界---障碍物、悬崖、风...
class GridMatrix(object):
    def __init__(self,n_width:int,
                      n_height:int,
                      default_type:int=0,
                      default_reward:float=0.0,
                      default_value:float=0.0):
        self.grids = None
        # assert (type(n_width)=="int")
        self.n_width = n_width # 水平方向格子数，int型
        self.n_height = n_height # 竖直方向格子数，int
        self.default_type = default_type # 默认类型
        self.default_reward = default_reward # 默认即时奖励值
        self.default_value = default_value # 默认价值

        self.len = n_width*n_height # 一维状态向量长度

        self.reset()

    def reset(self):
        """初始化"""
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,y,self.default_type,self.default_reward,self.default_value))

    """基本功能"""
    def get_grid(self,x,y=None):
        """
        获取一个格子的信息
        :param x: x坐标信息
        :param y: y坐标信息
        注意，坐标信息，由x,y表示或者仅有一个类型为tuple的x表示
        :return: grid object
        """
        xx,yy = None,None
        if isinstance(x,int): #
            xx,yy = x,y
        elif isinstance(x,tuple):
            xx,yy = x[0],x[1]
        # 坐标区域限制 --- 合理区间
        assert (xx>=0 and y>=0 and xx<self.n_width and yy<self.n_height)
        index = yy*self.n_width + xx # 索引
        return self.grids[index]

    def set_reward(self,x,y,reward):
        """设置奖励格子"""
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.reward = reward
        else:
            raise("你查找的这个Grid不存在")

    def get_reward(self,x,y):
        grid = self.get_grid(x,y)
        if grid is None:
            return None
        return grid.reward

    def set_value(self,x,y,value):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.value = value
        else:
            raise("你查找的这个Grid不存在")

    def set_type(self,x,y,type):
        """设置不同类型格子"""
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.value = type
        else:
            raise("你查找的这个Grid不存在")

    def get_value(self,x,y):
        grid = self.get_grid(x,y)
        if grid is None:
            return None
        return grid.value

    def get_type(self,x,y):
        grid = self.get_grid(x,y)
        if grid is None:
            return None
        return grid.type

# 强化学习环境---格子世界(各种各样的)
class GridWorldEnv(gym.Env):
    """格子世界环境，可以模拟不同的格子世界，包括随机行走、
    悬崖行走、风力干扰"""
    metadata = {
        'render.modes':['human','rgb_array'],
        'video.frames_per_second':30
    }

    def __init__(self,n_width:int=10,
                      n_height:int=7,
                      u_size = 40,
                      default_reward:float=0.0,
                      default_type = 0,
                      windy=False):
        self.u_size = u_size # 当前格子的尺寸
        self.n_width = n_width # 宽格子数
        self.n_height = n_height # 高格子数
        self.width = u_size * n_width # 场景宽度
        self.height = u_size * n_height
        self.default_reward = default_reward
        self.default_type = default_type

        self._adjust_size()  # 更新格子尺寸

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)

        self.reward = 0 # 用于可视化
        self.action = None # 用于可视化

        self.windy = windy # 是否加上风的影响

        # 定义动作空间
        self.action_space = spaces.Discrete(4) #上下左右
        # 定义状态空间
        self.observation_space = spaces.Discrete(self.n_height*self.n_width)

        # 起始点 左下角
        # 终点，<7,3>
        # 设置不同起始点、终点、奖励、格子类型，模拟不同世界
        # 随机行走、汽车租赁、悬崖行走...
        self.ends = [(7,3)] # 终止格子坐标
        self.starts = (0,3) # 起始点  后期若研究多智能体，再设置为列表
        self.types = [] # 特殊种类格子
        self.rewards = [] #特殊奖励格子
        self.refresh_setting()

        self.seed()
        self.viewer = None # 可视化接口
        self.reset()


    # 用于刷新格子世界的种类
    def refresh_setting(self):
        for x,y,r in self.rewards:
            self.grids.set_reward(x,y,r)
        for x,y,t in self.types:
            self.grids.set_type(x,y,t)

    def _adjust_size(self):
        """调整场景尺寸适合最大宽度,maxmize=40*20=800"""
        pass

    def _windy_effect(self,x,y):
        """加入风力干扰"""
        new_x,new_y = x,y
        if self.windy: # 是否要加风..?
            if new_x in [3,4,5,8]:
                new_y += 1
            elif new_x in [6,7]:
                new_y += 2
        return new_x,new_y

    def _action_effect(self,x,y,action):
        new_x,new_y = x,y
        if action == 0:
            new_x -= 1 # left
        elif action == 1:
            new_x += 1 # right
        elif action == 2:
            new_y += 1 # up
        elif action == 3:
            new_y -= 1 # down

        # 斜着
        elif action == 4: #左下
            new_x,new_y = new_x-1,new_y-1
        elif action == 5: #右上
            new_x,new_y = new_x+1,new_y+1
        elif action == 6: #右下
            new_x,new_y = new_x+1,new_y-1
        elif action == 7: #左上
            new_x,new_y = new_x-1,new_y+1

        return new_x,new_y

    def _boundary_effect(self,x,y):
        """悬崖"""
        new_x = x
        new_y = y
        if new_x < 0:
            new_x = 0
        if new_x >= self.n_width:
            new_x = self.n_width - 1
        if new_y < 0:
            new_y = 0
        if new_y >= self.n_height:
            new_y = self.n_height-1
        return new_x,new_y

    #------ 老四样 -------
    def step(self,action):
        assert self.action_space.contains(action),\
            "%r (%s) invalid" % (action,type(action))

        self.action = action # 可视化用的action

        old_x,old_y = self._state_to_xy(self.state)
        new_x,new_y = old_x,old_y

        # windy effect
        # 若此时智能体在有风的格子内，会被吹离
        new_x,new_y = self._windy_effect(new_x,new_y)
        # 行为效果(执行action)
        new_x,new_y = self._action_effect(new_x,new_y,action)
        # 边界约束
        new_x,new_y = self._boundary_effect(new_x,new_y)
        # 障碍物 type = 1 是障碍物
        if self.grids.get_type(new_x,new_y) == 1:
            new_x ,new_y = old_x, old_y  # 回退
        # 奖励函数麻烦给一下
        self.reward = self.grids.get_reward(new_x,new_y)
        done = self._is_end_state(new_x,new_y)
        # 在__init__中，
        self.state = self._xy_to_state(new_x,new_y)
        # 提供格子世界所有的信息在info内
        info = {"x":new_x,"y":new_y,"grids":self.grids}
        return self.state,self.reward,done, info

    # 图形化窗口
    def render(self, mode='human',close=False):
        """图形化"""
        """
        Viewer里绘制集合图像的基本步骤：
        --- 1. 建立该对象需要的数据本身
        --- 2. 使用rendering提供的方方返回一个geom对象
        --- 3. 对geom对象进行如颜色、线宽...属性设置，最终要的是变换属性：
        --- --- 3.1 变换属性: 负责对对象在屏幕中的位置、渲染、缩放进行渲染；若某对象需要进行上述变化，建立变换属性
        --- --- 3.2 变换属性是一个Transform对象，包括 translate、rotate、scale三个属性, 每个属性都以np.array对象描述的矩阵决定
        --- 4. 将建立的geom对象添加至viewer的绘制对象列表中，屏幕上出现一次，就add_onegeom(),多次就add_geom()
        --- 5. 渲染整个viewer前，对有需要的geom的参数进行修改，主要基于Transform对象;
        --- 6. 调用Viewer的render()方法进行绘制
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # ----- 基本参数 -----
        zero = (0,0)
        u_size = self.u_size # 当前格子的尺寸
        m = 2 # 格子之间间隙尺寸

        # 若未设定屏幕对象，初始化整个屏幕具备的元素
        if self.viewer is None:
            from gym.envs.classic_control import rendering # 经典控制风格
            self.viewer = rendering.Viewer(self.width,self.height)

            # 绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):

                    # 建立对象数据本身
                    v = [(x*u_size+m,y*u_size+m),
                         ((x+1)*u_size-m,y*u_size+m),
                         ((x+1)*u_size-m,(y+1)*u_size-m),
                         (x*u_size+m,(y+1)*u_size-m)] # 矩形四点坐标

                    # rendering返回geom对象
                    rect = rendering.FilledPolygon(v)

                    # 修改geom属性
                    r = self.grids.get_reward(x,y)/10 # 归一化
                    if r < 0:
                        rect.set_color(0.9-r,0.9+r,0.9+r) # 负奖励格子颜色
                    elif r > 0:
                        rect.set_color(0.3,0.5+r,0.3) # 正奖励格子颜色
                    else:
                        rect.set_color(1.0,0.8,1.0) # 普通

                    # 添加到viewer绘制对象中
                    self.viewer.add_geom(rect)
                    # 绘制边框
                    v_outline = [(x*u_size+m,y*u_size+m),
                         ((x+1)*u_size-m,y*u_size+m),
                         ((x+1)*u_size-m,(y+1)*u_size-m),
                         (x*u_size+m,(y+1)*u_size-m)] # 矩形四点坐标
                    outline = rendering.make_polygon(v_outline,False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x,y):
                        # 终点方格添加金色
                        outline.set_color(0.9,0.9,0)
                        self.viewer.add_geom(outline)
                    if self.starts[0] == x and self.starts[1] == y:
                        outline.set_color(0.5,0.5,0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x,y) == 1: # 障碍格子，深灰色
                        rect.set_color(0.3,0.3,0.3)
                    else:
                        pass

            # 绘制智能体
            self.agent = rendering.make_circle(u_size/4,30,True)
            self.agent.set_color(1.0,1.0,0.0)
            self.viewer.add_geom(self.agent)

            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        # 更新智能体位置
        x,y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x+0.5)*u_size,(y+0.5)*u_size)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def reset(self):
        self.state = self._xy_to_state(self.starts)
        return self.state

    def seed(self,seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]

    # ------ 工具函数 ------
    def _state_to_xy(self,s):
        """一维状态量(当前所在格子号)转换为x,y"""
        x = s % self.n_width # 当前格子数/宽度的余数即x坐标
        y = int((s-x)/self.n_width) #当前格子数-x/宽度即y坐标
        return x,y

    def _xy_to_state(self,x,y=None):
        """x,y转变为状态量，注意x可能是一个标量，也可能是包含[x,y]的向量"""
        if isinstance(x,int):
            assert (isinstance(y,int)), "incomplete Position info"
            return x+self.n_width*y
        elif isinstance(x,tuple):
            return x[0] + self.n_width*x[1]

    def _is_end_state(self,x,y=None):
        """
        判断是否是终点，以设置done值
        输入可以是any form
        """
        if y is not None: #按照x,y坐标给
            xx,yy = x,y
        elif isinstance(x,int):
            xx,yy = self._state_to_xy(x) # 按照状态量给
        else:
            assert(isinstance(x,tuple)), "坐标数据不完整"
            xx,yy = x[0],x[1] # 按照元组给
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False


# if __name__ == "__main__":
#     # 测试
#     env = GridWorldEnv()
#     print("Hello, Welcome")
#     env.reset()
#     nfs = env.observation_space
#     nfa = env.action_space
#     print(env.observation_space)
#     print(env.action_space)
#     print(env.state)
#
#     env.render()
#     for _ in range(200):
#         # 随机采样
#         env.render()
#         a = env.action_space.sample() # 随机采样
#         state,reward,isdone,info = env.step(a)
#
#     print("env.closed")

