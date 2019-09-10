#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   plot_images.py
@time    :   2019/09/07 19:46:45
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import cv2
import matplotlib.pyplot as plt
import math

#%%
def plot_images(images, idx_fig=1, figsize=(12, 24)):
    """
    """
    fig = plt.figure(num=idx_fig, figsize=figsize)
    fig.clf()

    num_images = len(images)
    num_cols = int(math.sqrt(num_images))
    num_rows = int(math.ceil(num_images / num_cols))

    idx_image = 0

    for idx_row in range(num_rows):

        for idx_col in range(num_cols):

            ax = fig.add_subplot(num_rows, num_cols, idx_image + 1,
                                 frameon=False)
            if images[idx_image].dtype == "uint8":
                ax.imshow(cv2.cvtColor(images[idx_image], cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(images[idx_image])
            ax.axis("off")

            idx_image += 1
            if idx_image >= num_images:
                break

    plt.show()


#%%
if __name__ == "__main__":

    image = cv2.imread("./template/ip_man_3.jpg")
    images = [image] * 10

    plot_images(images)

#%%
