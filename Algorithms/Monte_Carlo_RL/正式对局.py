#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from 决胜21点 import *
from 赌桌环境 import *
# from utils import set_dict,get_dict
from 工具函数 import *

# 定义动作空间
A = ["叫牌","停止叫牌"]
display = False
# 创建玩家
player = Player(A=A,display=display)
# 创建庄家
dealer = Dealer(A=A,display=display)
# 创建赌桌
arena = Arena(A=A,display=display)
# 生成num个完整的对局，并打印对局信息
arena.play_games(dealer,player,num=200000)

"""
1.未引入MC学习策略
###############
一共玩了2000局
玩家赢618局,输95局,平1287局;
胜率:0.31
###############
"""