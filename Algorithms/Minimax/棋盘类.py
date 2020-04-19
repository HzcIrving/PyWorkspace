#! /usr/bin/enc python
# -*- coding: utf-8 -*-

"""
三子棋游戏
模式选择：
--- 1. player vs player
--- 2. AI vs AI
--- 3. player vs AI
电脑玩家使用了minimax算法，带alpha-beta剪枝算法
电脑玩家在思考时候，时刻思考"假想敌"，以运转minmax算法
原blog:https://www.cnblogs.com/hhh5460/p/7082112.html
"""

import numpy as np
import math
import time

class Board(object):
    """定义棋盘"""
    def __init__(self):
        self._board = ["-" for _ in range(9)]
        self._history = [] # 保存下过的棋谱

    def _move(self,action,take):
        """按指定动作放入棋子"""
        if self._board[action]=="-": # 检查是否已经落子
            self._board[action] = take
            self._history.append((action,take))

    def _unmove(self,action):
        self._board[action] = "-"
        self._history.pop() # 移除列表中最后一个元素

    def get_board_snapshot(self):
        """输出当前棋盘信息"""
        return self._board[:]

    def get_legal_actions(self):
        """选择合法动作进行落子"""
        actions = []
        for i in range(9):
            if self._board[i] == '-':
                actions.append(i)
        return actions

    def is_legal_action(self,action):
        """判断走法是否合法，是否在'-'上落子"""
        return self._board[action] == '-'

    def terminate(self):
        """终局检测"""
        board = self._board
        # a [1,2,3,4,5,6,7,8,9]
        # a[0::3] -- > [1,4,7] #从0开始每隔3取一个
        # a[1::3] -- > [2,5,8]
        # a[0::4] -- > [1,5,9] # 对角
        # a[2:7:2] -- > 从3到7,每两个取一个，列表基本操作
        self.lines = [board[0:3],board[3:6],board[6:9],board[0::3],board[1::3],board[2::3],board[0::4],board[2:7:2]]
        if ['X']*3 in self.lines or ['O']*3 in self.lines or '-' not in board:
            # 终局检测，3点一线
            return True
        else:
            return False

    def win_or_lose(self):
        """胜负检查"""
        board = self._board
        lines = self.lines
        if ['X']*3 in lines:
            return 0 # 0代表"X"胜利
        elif ['O']*3 in lines:
            return 1 # 1代表"O"胜利
        else:
            return 2 # 2代表平局

    def print_board(self):
        """打印棋盘"""
        board = self._board
        for i in range(len(board)):
            print(board[i],end='')
            if (i+1)%3 == 0:
                print() # 换行

    def print_history(self):
        print("本局棋谱：",self._history)




