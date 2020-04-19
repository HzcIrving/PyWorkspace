#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
学习Python ArgParse
Python 自带命令行参数解析包
-- 1. 方便读取命令行参数
-- 2. 分离参数和代码（当你的代码需要频繁修改参数时，可以使用这个工具）
"""

# DEMO 1 basic structure
import argparse

def get_parser():
    """从命令行获取用户名，然后打印'Hello+用户名'"""
    # 1.生成参数解析器对象
    parser = argparse.ArgumentParser(description="Demo of Argparse")
    # 2.add_argument函数来增加参数,default参数表示如果没提供参数，我们默认采用的值
    parser.add_argument('--name',default='Great',help='请输入姓名') #  默认为great
    # 3.required 表示这个参数是否一定需要设置
    parser.add_argument('--sex',required=True)
    # 4.type:参数类型
    parser.add_argument('--num',type=int)
    parser.add_argument('--bool',type=bool)
    # 5.choices: 参数只可以从几个选项里面进行选择
    parser.add_argument('--handb',required=True,choices=['hzc','llf'])


    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # args.sex = 'girl'
    name = args.name
    sex = args.sex
    res = args.num**2
    bool = args.bool

    print('Hello {}'.format(name))
    print('Sex {}'.format(sex))
    print('Num^2 {}'.format(res))
    print('Bool {}'.format(bool))
    print('Handsome Guy {}'.format(args.handb))

# 在命令行中修改
# $ python print_name.py - -name Wang
# Hello  Wang

# python ArgParse.py --name Wang --sex Girl
# Hello Wang
# Sex Girl

# ... 同理

# python ArgParse.py -h 可以查看参数信息

