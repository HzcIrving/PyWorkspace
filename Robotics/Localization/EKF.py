#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

"""
基于扩展Kalman Filter的EKF的定位算法
EKF --- 是标准卡尔曼滤波在非线性情况下的一种扩展形式
参考链接：https://blog.csdn.net/weixin_42647783/article/details/89054641

Basic---利用Taylor展开，将非线性系统线性化，采用Kalman滤波框架对信号进行滤波
所以是一种次优滤波

EKF在对非线性函数做泰勒展开时，只取到一阶导数和二阶导数；

Pcode:
1. Set initial Value ... xEst,PEst
2. Predict state & error covariance ... xpred,Ppred
3. Compute Kalman Gain ... K 反应两个误差矩阵的偏重
4. Compute the estimate ... xEst
5. Compute the error covariance ... PEst
6. Return to 1.
"""

# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math

# Robot状态向量 x,y,phi,v_t
# xEst --- state vector
# input : v_t,w_t

# EKF simulation的协方差矩阵初始化
Q = np.diag([
    0.1, # x-axis的定位误差
    0.1, # y-axis的定位误差 variance of location
    np.deg2rad(1.0), # yaw angle的误差
    1.0 # 速度误差
])

R  = np.diag([1.0,1.0]) ** 2 # x,y的postion的观测误差

# 仿真参数
INPUT_NOISE = np.diag([1.0,np.deg2rad(30.0)]) ** 2 # 输入噪声
GPS_NOISE = np.diag([0.5,0.5])**2 # GPS噪声
DT = 0.1 # 时间参数
SIM_TIME = 100.0 # 仿真时间

show_animation = True

def calc_input():
    """输入函数"""
    v = 1.0 #m/s
    yawrate = 0.1 #rad/s
    u = np.array([[v],[yawrate]])
    return u

def ekf_estimation(xEst,PEst,z,u):
    """
    扩展Kalman Filter的定位过程
    ===Predict===
    x_pred = Fx_t + Bu_t
    P_pred = J_F P_t J_F.T + Q
    ===Update===
    z_pred = Hx_pred
    y = z - z_pred
    S = J_H P_pred J_H.T + R
    K = P_pred J_H.T S^{-1}
    x_{t+1} = x_pred + Ky
    P_t+1 = (I-KJ_H)_P_pred
    """
    #  预测
    xPred = motion_Model(xEst,u)
    jF = jacob_f(xPred,u) # 运动模型的雅克比矩阵
    PPred = jF@PEst@jF.T + Q

    # Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

    return xEst,PEst



def observation(xTrue,xd,u):
    """
    观测函数
    :param xTrue: Robot本该在的位置
    :param xd: 实际轨迹
    :param u: 输入
    :return:
    """
    xTrue = motion_Model(xTrue,u) #更新期望状态信息
    # 给GPS加入噪声
    z = observation_model(xTrue)+GPS_NOISE@np.random.randn(2,1)
    # 给输入加噪
    ud = u + INPUT_NOISE@np.random.randn(2,1)

    xd = motion_Model(xd,ud) # 实际状态信息
    return xTrue,z,xd,ud

def motion_Model(x,u):
    """
    运动模型
    状态--x,y,phi,v
    input--v,omega
    x' = vcos(phi)
    y' = vsin(phi)
    phi' = omega
    --> x_{t+1} = Fx_t + Bu_t
    F = [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0].
         [0,0,0,0]]
    B = [[cos(phi)dt,0],
         [sin(phi)dt,0],
         [0,dt],
         [1,0]]
    """
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0], #x[2,0]--phi
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
    x = F@x+B@u
    return x

def observation_model(x):
    """
    观测模型，Robot通过GPS传感器可以获得x,y的位置信息
    x:输入状态信息
    :return: 返回位置信息
    Z_t = Hx_t
    """
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z

def jacob_f(x,u):
    """
    运动模型的Jacobian矩阵
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return jF

def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH

# def plot_covariance_ellipse(xEst,PEst)

def main():
    print(__file__+" start!!!")

    time = 0.0

    # State Vector [x,y,yaw,v]
    xEst = np.zeros((4,1))
    xTrue = np.zeros((4,1))
    PEst = np.eye(4)

    xDR = np.zeros((4,1)) # 航位推算 Dead Reckoning

    # Dead Reckoning推算，是指若不进行Kalman滤波，则实际的
    # 运行轨迹会是多少

    # history历史信息
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2,1)) # 初始化观测

    while SIM_TIME>= time:
        time += DT
        u = calc_input()
        xTrue,z,xDR,ud = observation(xTrue,xDR,u)

        xEst,PEst = ekf_estimation(xEst,PEst,z,ud) # Kalman滤波定位实际位置

        # store历史信息
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            #通过"esc"键来退出仿真
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda  event: [exit(0) if event.key=='escape' else None])
            plt.plot(hz[0,:],hz[1,:],".g",label="Observation")
            plt.plot(hxTrue[0,:].flatten(),
                     hxTrue[1,:].flatten(),"-b",label="Desired")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k",label="Dead Reckon")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r",label="Estimation")

            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
    # plt.show()


if __name__ == '__main__':
    main()
    # plt.show()
