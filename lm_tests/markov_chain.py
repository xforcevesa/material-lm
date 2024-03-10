import numpy as np
import pylab as pl

p01 = np.array([0.5, 0.2, 0.3])  # p0数据 1，矩阵中每一行概率之和总等于1
p02 = np.array([0.1, 0.4, 0.5])  # p0数据 2， 两个初始状态，用于分别计算和画图
p = np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])
n = 30
c = np.array(['r', 'g', 'b'])  # 定义画图颜色数组


def calanddraw(p0, p):  # 2 计算并画图函数
    for i in range(n):  # 3
        p0 = np.mat(p0) * np.mat(p)  # 迭代变量 确定p0与p 的关系
        for j in range(len(np.array(p0)[0])):
            pl.scatter(i, p0[0, j], c=c[j], s=.5)  # 确定画点属性


pl.subplot(121)
calanddraw(p01, p)  # 调用数据画图
pl.subplot(122)
calanddraw(p02, p)  # 调用数据画图
pl.show()
