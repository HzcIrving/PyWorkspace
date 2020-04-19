#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
基于MC学习评估二十一点游戏

----游戏规则----
1. 一个庄家(dealer), 一个玩家(player);
2. 一共52张扑克牌，2~10代表2~10点，JQK代表10点，A可以记为1点
   也可以记为11点;
3. 开局，庄家一次连续发2张牌给庄家和玩家，庄家第一张牌是明牌，
   牌面信息开放，后续发牌不开放，玩家可以根据手中的牌决定"Twist"
   继续叫牌，或者是"Stick"停止叫牌，玩家可以持续叫牌，但是当手
   牌中点数超过21点，停止叫牌；
4. 玩家停止叫牌后，庄家可以继续叫牌，若庄家停止叫牌，则公布手牌，
   判断输赢；
5. 计算输赢问题：
    5.1 --- 双方点数均超过21点，或者双方点数相同： 和局；
    5.2 --- 一方21点，另一方不是21点：点数为21点的游戏者胜；
    5.3 --- 双方点数均不到21点，点数离21点近的玩家赢。

6. 玩家是否有可用A的问题：
   玩家手牌 "A,3,6" 11+3+6 = 20 < 21 : 称为可用的A
   玩家手牌 "A,5,7" 11+5+7 = 23 > 21 : 称为不可用的A

----强化学习问题----
1. 状态空间(A,B,bool)
   比如 (10,15,0) 表示，庄家明牌点数是10，玩家手牌中点数15，无可用A;
   比如 (A,17,1) 表示是庄家明牌为A， 玩家手牌中点数17,有可用A;
2. 行为空间:
   继续叫牌 --- Twist
   停止叫牌 --- Stick
3. 庄家策略：
   初始---只有手中牌点数达到或者超过17，停止叫牌
4. 玩家策略：
   初始---只有手中牌点数不到20，继续叫牌，否则停止
5. 衰减因子\gamma = 1
"""

# 搭建游戏环境
from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from 工具函数 import *

# 游戏者共同属性 1.姓名，2.行为空间，3.是否在终端显示信息
# 游戏者共同特点：
#   1. 应该至少能记住对局过程中手中牌的信息；
#   2. 知道自己的行为空间;
#   3. 辨认单张牌的点数以及手中牌的总点数。
class Gamer():
    """游戏者---庄家、玩家的基本属性"""
    def __init__(self,name="",A=None,display=False):
        self.name = name
        # self.role = self.name
        self.cards = [] # 手中的牌
        self.display = display # 是否显示对局文字信息
        self.policy = None # 策略
        self.learning_method = None # 学习方法 # 后面用MC
        self.A = A # action space

    def __str__(self):
        return self.name  # 玩家姓名

    def _value_of(self,card):
        """
        根据牌的字符判断数值大小
        :param card: 牌面信息 如,list['A','10,'3']
        :return: 牌的大小数值
        """
        try:
            v=int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ["J","Q","K"]:
                v = 10
            else:
                v = 0
        finally:
            return v

    def get_points(self):
        """
        统计分值，若使用了A为1点，则同时返回True
        :return:
            tuple(返回牌总点数，是否使用了可复用的A)
            eg.['A','10','13'] 返回(14,False)
            eg.['A','10'] 返回(21,True)
        """
        num_of_useable_ace = 0 #默认没有拿到Ace
        total_point = 0
        cards = self.cards
        if cards is None:
            return 0,False
        for card in cards:
            v = self._value_of(card)
            if v == 1:
                num_of_useable_ace+=1
                v = 11
            total_point += v
        while total_point>21 and num_of_useable_ace>0:
            total_point-=10 # 若大于21，则非有效A
            num_of_useable_ace -= 1
        return total_point,bool(num_of_useable_ace)

    def receive(self,cards=[]):
        """
        添加牌
        """
        cards = list(cards)
        for card in cards:
            self.cards.append(card) # 添加牌

    def discharge_cards(self):
        """清空手中的牌"""
        self.cards.clear()

    def cards_info(self):
        """显示牌面具体信息"""
        self._info("{}{}现在的牌:{}\n".format(self.role,self,self.cards))

    def _info(self,msg):
        if self.display:
            print(msg,end="")


class Dealer(Gamer):
    """给Gamer添加庄家标签，指定策略"""
    def __init__(self,name="",A=None,display=False):
        super(Dealer,self).__init__(name,A,display)
        self.role="庄家" #角色
        self.policy = self.dealer_policy #庄家策略

    def first_card_value(self):
        """庄家第一张牌明牌"""
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])

    def dealer_policy(self,Dealer=None):
        """庄家策略的细节"""
        action = ""
        dealer_points,_ = self.get_points()
        if dealer_points>=17:
            action = self.A[1] #停止叫牌
        else:
            action = self.A[0] #继续叫牌
        return action

class Player(Gamer):
    """给Gamer添加玩家标签，指定策略"""
    def __init__(self,name="",A=None,display=False):
        super(Player,self).__init__(name,A,display)
        self.policy = self.naive_policy  # 玩家策略(1）
        self.role = "玩家"

    def get_state(self,dealer):
        """
        状态空间---(dealer的明牌,自己的牌,是否有可用的ace)
        """
        dealer_first_card_value = dealer.first_card_value()
        player_points,useable_ace = self.get_points()
        return dealer_first_card_value,player_points,useable_ace

    def naive_policy(self,dealer=None):
        """玩家策略，看是否大于20点数决定是否叫牌"""
        player_points,_ = self.get_points()
        if player_points < 20:
            action = self.A[0]
        else:
            action = self.A[1]
        return action

    def get_state_name(self,dealer):
        return str_key(self.get_state(dealer))




