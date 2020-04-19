#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from 决胜21点 import *
# from utils import  *
from 工具函数 import *

# 准备游戏环境
# 功能如下:
#     1. 游戏桌
#     2. 游戏牌
#     3. 组织游戏对局 --- 洗牌，发牌
#     4. 判定输赢 --- 根据庄家Dealer和玩家Player手牌信息，判断输赢
#     5. 回收每回合的废牌
class Arena():
    """游戏环境"""
    def __init__(self,display=None,A=None):
        self.cards = ['A','2','3','4','5','6','7','8','9','10','J',\
                      'Q','K']*4 # 52张牌
        self.card_q = Queue(maxsize=52) # 洗好的牌 定义一个FIFO队列
        self.cards_in_pool = [] # 用来存放已经用过的公开的牌---牌池
        self.display = display
        self.episodes = [] # 产生的对局信息列表
        self.load_cards(self.cards)
        self.A = A # action space

    # 将cards_in_pool传入，即桌面上已经使用废牌收集，然后洗牌
    def load_cards(self,cards):
        """把收集的牌进行洗牌，装入发牌器中"""
        shuffle(cards)
        for card in cards: # deque只能一个一个添加
            self.card_q.put(card)
        cards.clear() # 原来的牌清空
        return

    # 判断输赢，给出奖励
    def reward_of(self,dealer,player):
        """判断输赢，给出奖励值"""
        dealer_points,_ = dealer.get_points()
        player_points,useable_ace = player.get_points()
        if player_points>21:
            reward = -1
        else:
            if player_points>dealer_points or dealer_points>21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward,player_points,dealer_points,useable_ace

    # 发牌
    def serve_card_to(self,player,n=1):
        """
        像庄家/玩家发出一定数量的牌；在发牌时，若发牌器里没有牌，则会
        把牌池里的牌收集起来洗好送入发牌器，然后把需要数量的牌发给某一
        玩家。
        :param player: 庄家或者玩家
        :param n: 一次连续发牌的数量
        :return: None
        """
        cards = [] # 将要发出的牌
        for _ in range(n):
            # 判断发牌器是否有牌了
            if self.card_q.empty():
                self._info("\n发牌器没牌了，整理废牌，重新洗牌;")
                shuffle(self.cards_in_pool)
                self._info("一共整理了{}张已用牌，重新放入发牌器\n".format(\
                    len(self.cards_in_pool)))
                assert(len(self.cards_in_pool)>20)

                self.load_cards(self.cards_in_pool) # 重新放入发牌器

            # 从发牌器发出一张牌(队列结构)
            cards.append(self.card_q.get())

        self._info("发了{}张牌({})给{}{};\n".format(n,cards,player.role,player))
        player.receive(cards) # 某玩家接受发出的牌
        player.cards_info()

    # 信息显示
    def _info(self,message):
        if self.display:
            print(message,end="")

    # 每个回合结束后回废牌至牌池
    def recycle_cards(self,*players):
        if len(players)==0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards() # 玩家手里不在保留，牌列表清空

    # 庄家和玩家进行一次对局
    def play_game_one_time(self,dealer,player):
        """
        玩一局21点，并生成状态序列和最终奖励(中间奖励为0)
        Returns:
            tuple:episode(状态序列),reward(奖励)
        """
        self._info("=====开始新的一局=====\n")
        # 初始化状态序列(对局信息)
        episode=[]

        # 发两张牌给玩家
        self.serve_card_to(player,n=2)
        # 发两张牌给庄家
        self.serve_card_to(dealer,n=2)

        if player.policy is None:
            self._info("玩家需要一个策略\n")
            return
        if dealer.policy is None:
            self._info("庄家需要一个策略\n")
            return

        while True:
            # Loop直到一方赢
            action = player.policy(dealer) # 产生行为
            # 玩家策略产生一个行为
            self._info("{}{}选择:{};".format(player.role,player,action))

            episode.append((player.get_state_name(dealer),action))  # 记录一个(s,a)

            if action == self.A[0]: # 叫牌
                self.serve_card_to(player) # 发一张牌给玩家
            else: # 停止叫牌
                break

        # 玩家停止叫牌后，计算下玩家手中的点数，若玩家"暴雷"，庄家win，不要继续了
        reward,player_points,dealer_points,useable_ace = self.reward_of(dealer,player)

        if  player_points>21:
            self._info("玩家爆点{}输了,得分:{}\n".format(player_points,reward))
            self.recycle_cards(player,dealer)
            self.episodes.append((episode,reward)) # 存入本局对局信息
            self._info("=====本局结束=====\n")
            return episode,reward

        # 玩家没有超过21点
        self._info("\n")
        while True:
            action = dealer.policy() # 庄家从策略中获取行为
            # 状态只记录庄家第一张牌面信息，此时玩家不在叫牌，无需重复记录
            if action == self.A[0]: # 庄家继续叫
                self.serve_card_to(dealer)
            else:
                break

        # 双方停止叫牌
        self._info("双方停止叫牌;\n")
        reward,player_points,dealer_points,useable_ace=self.reward_of(dealer,player)
        # 显示双方牌面信息
        player.cards_info()
        dealer.cards_info()

        if reward == +1:
            self._info("玩家赢了(*^▽^*)")
        elif reward == -1:
            self._info("玩家输了(⊙︿⊙)")
        else:
            self._info("双方和局ε=(´ο｀*)))唉")
        self._info("玩家{}点，庄家{}点\n".format(player_points,dealer_points))
        self._info("=====本局结束=====\n")
        self.recycle_cards(player,dealer) # 回收牌
        self.episodes.append((episode,reward)) # 将刚才的完整对局信息添加状态序列列表
        return episode,reward

    def play_games(self,dealer,player,num=2,show_statistic=True):
        """
        一次性玩多局游戏,接受一个庄家，一个玩家，需要产生的对局数量，以及是否
        显示多个对局的统计信息
        :param dealer: 庄家
        :param player: 玩家
        :param num: 玩的局数
        :param show_statistic: 是否显示数据
        :return: None
        """
        results = [0,0,0] # 玩家负、和、胜利局数
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode,reward = self.play_game_one_time(dealer,player)
            results[1+reward]+=1

            # 每次结束后，若玩家能够从中学习，则提供玩家一次学习机会
            if player.learning_method is not None:
                player.learning_method(episode,reward)

        if show_statistic:
            print("###############\n一共玩了{}局\n玩家赢{}局,输{}局,平{}局;\n胜率:{:.2f}\n###############"\
                  .format(num,results[2],results[1],results[0],results[2]/num))
        return


