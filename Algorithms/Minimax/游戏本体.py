#! /usr/bin/enc python
# -*- coding: utf-8 -*-

from 棋盘类 import Board
from 玩家 import *

class GAME(object):
    def __init__(self):
        self.board = Board() # 棋盘
        self.current_player = None

    # 生产玩家
    def mk_player(self,p,take='X'):
        if p == 0 :
            return HumanPlayer(take) # 人类玩家
        else:
            return AIPlayer(take) # AI玩家

    # 交换玩家 (轮流下棋)
    def switch_player(self,player1,player2):
        if self.current_player == None:
            return player1
        else:
            return [player1,player2][self.current_player==player1]

    # 打印赢家
    def print_winner(self,winner):
        print(['胜利者是player1','胜利者是player2','平局'][winner])

    # 运行游戏
    def run(self):
        ps = input("请选择两个玩家类型: \n\t0.人类玩家\n\t1.AI\n输入方式:0 0\n")
        p1,p2 = [int(p) for p in ps.split(' ')]
        player1,player2 = self.mk_player(p1,'X'),self.mk_player(p2,'O') # 先手执X，后手执O

        print("\n游戏开始...\n")
        self.board.print_board() # 显示棋盘
        while True:
            self.current_player = self.switch_player(player1,player2) # 交换
            action = self.current_player.think(self.board) # 当前玩家对棋局思考，得到落子方法
            self.current_player.move(self.board,action) # 执行，改变棋盘
            self.board.print_board() # 显示棋盘
            if self.board.terminate(): # 判断棋盘是否终止
                winner = self.board.win_or_lose() # 得到赢家
                break
        self.print_winner(winner)
        print("游戏结束")
        self.board.print_history()

if __name__ == '__main__':
    GAME().run()