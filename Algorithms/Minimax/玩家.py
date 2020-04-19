#! /usr/bin/enc python
# -*- coding: utf-8 -*-

"""
玩家的行为很简单：先思考，得到走法，后直接落子，棋盘相应改变即可
"""
from 棋盘类 import Board

class Player(object):
    """玩家"""
    def __init__(self,take='X'):
        """人类玩家执'X'"""
        self.take = take

    def think(self,board):
        pass

    def move(self,board,action):
        board._move(action,self.take)

class HumanPlayer(Player):
    """人类玩家"""
    def __init__(self,take):
        super().__init__(take) # 继承，默认执"X"

    def think(self,board):
        while True:
            action = input("请输入0~8的数字")
            if len(action)==1 and action in '012345678' and board.is_legal_action(int(action)):
                """检测合法性,在9个格子中选一个"""
                return int(action)
            else:
                print("错误，非法输入，请重新选择")

class AIPlayer(Player):
    """电脑"""
    def __init__(self,take):
        super().__init__(take)

    def think(self,board):
        print("等待对手落子...")
        take = ['X','O'][self.take=='X'] # 若take是X,此时take取O,否则取X
        player = AIPlayer(take) # 假想的敌人
        _,action =self.minimax(board,player)
        return action

    def minimax(self,board,player,depth=0):
        """
        Minimax算法本体
        :param board: 棋盘
        :param player: 玩家
        :param depth: 博弈树深度
        :return: bestVal, bestAction
        """
        if self.take == "O":
            bestVal = -10
        else:
            bestVal = 10

        if board.terminate():
            if board.win_or_lose() == 0:
                # "X"胜利
                return -10 + depth, None
            elif board.win_or_lose() == 1:
                # "O"胜利
                return 10 - depth, None
            elif board.win_or_lose() == 2:
                # 平局
                return 0,None

        # 遍历合法走法
        for action in board.get_legal_actions():
            board._move(action,self.take)
            val,_ = player.minimax(board,self,depth+1) # 切换到假想敌
            board._unmove(action) # 撤销走法，回溯

            if self.take == "O":
                if val > bestVal: # Max
                    bestVal,bestAction = val,action
            else: # Min
                if val < bestVal:
                    bestVal,bestAction = val,action
        return bestVal,bestAction


