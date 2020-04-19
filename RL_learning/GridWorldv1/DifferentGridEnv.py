#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
多种grid环境库
陆续更新
1---WindyGridWorld 有风 10x7
"""

from GridWorldv1 import *

def WindyGridWorld():
    env = GridWorldEnv(
        n_width=10, n_height=7,u_size=40,default_type=0,default_reward=-1,windy=True
    )
    env.starts = (0,3)
    env.ends = [(7,3)]
    env.rewards = [(7,3,0)] # 终点处奖励为0，其余为-1

    env.refresh_setting()
    return env

# 悬崖行走格子环境
class CliffWalk2(GridWorldEnv):
    def __init__(self,n_width=12,n_height=4,u_size=40,default_reward=-1,\
                 default_type=0,windy=False):
        super(CliffWalk2,self).__init__(n_width=12,n_height=4,u_size=80,default_reward=-1,default_type=0,windy=False)
        self.starts = (0,0)
        self.ends = [(11,0)]
        self.rewards = [(11,0,0)]
        for i in range(10):
            self.rewards.append((i+1,0,-100)) # 悬崖处单步为-100的惩罚
        self.refresh_setting()

    def step(self,action):
        """重写悬崖的step"""
        assert self.action_space.contains(action),\
            "%r (%s) invalid" % (action,type(action))
        self.action = action

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

        self.reward = self.grids.get_reward(new_x,new_y)

        # ------------修改部分--------------
        if 0<new_x<11 and new_y == 0:
            new_x,new_y = 0,0  # 到悬崖了 任务失败

        done  = self._is_end_state(new_x,new_y)
        self.state = self._xy_to_state(new_x,new_y)
        # 提供格子世界所有的信息
        info = {"x":new_x,"y":new_y,"grids":self.grids}
        return self.state,self.reward,done, info


# if __name__ == '__main__':
#     # windyenv = WindyGridWorld()
#     # print("欢迎来到有风的格子环境")
#     #
#     # env = CliffWalk2()
#     # print("欢迎来到悬崖环境")
#     # # windyenv.reset()
#     # env.reset()
#     #
#     # # windyenv.render()
#     # env.reset()
#     # for _ in range(200):
#     #     env.render()
#     #     a = env.action_space.sample()
#     #     start,reward,done,info = env.step(a)
#     #
#     # print("env.closed")
