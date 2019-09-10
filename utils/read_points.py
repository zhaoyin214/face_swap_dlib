#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   read_points.py
@time    :   2019/09/08 13:07:14
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import os
import pandas as pd


#%%
def read_points(filepath):
    """
    reading points
        x y
        x y
        x y
        ...
        x y

    arguments:
        filepath (str) - point file path

    return:
        points (pd.dataframe)
    """
    assert os.path.isfile(filepath), "ERROR: file does not exist!"
    with open(file=filepath, mode="r", encoding="utf-8") as fr:
        lines = fr.readlines()

    points = []
    for line in lines:
        points.append([int(item) for item in line.strip().split(" ")])
    points = pd.DataFrame(points, columns=["x", "y"])

    return points


#%%
if __name__ == "__main__":

    from configs import TEMPLATE_LANDMARKS_PATH

    points = read_points(TEMPLATE_LANDMARKS_PATH)
    print(points)

#%%
