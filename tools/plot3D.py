# -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import importlib
importlib.import_module('mpl_toolkits').__path__
from mpl_toolkits.mplot3d import Axes3D
def draw_3d(xs, ys, zs, x_label_name, y_label_name, z_label_name):
    ax = plt.figure().add_subplot(111, projection='3d')
    # 基于ax变量绘制三维图
    # xs表示x方向的变量
    # ys表示y方向的变量
    # zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
    # m表示点的形式，o是圆形的点，^是三角形（marker)
    # c表示颜色（color for short）
    ax.scatter(xs, ys, zs, c='r', marker='^')  # 点为红色三角形

    # 设置坐标轴
    ax.set_xlabel(x_label_name)
    ax.set_ylabel(y_label_name)
    ax.set_zlabel(z_label_name)

    # 显示图像
    plt.show()
if __name__ == '__main__':
    xs = [1 + 0.01 * i for i in range(1000)]
    ys = [1 + 0.01 * i for i in range(1000)]
    zs = [1 + 0.01 * i for i in range(1000)]
    draw_3d(xs, ys, zs)