#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   faceswap_image.py
@time    :   2019/09/06 19:10:03
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

import sys
sys.path.append("e:/src/jupyter/cv/face_swap_dlib")

#%%
import cv2
import numpy as np
import pandas as pd

from configs import TRIANGULATION_COLUMES, \
    CV_RECT_XMIN, CV_RECT_YMIN, CV_RECT_WIDTH, CV_RECT_HEIGHT, \
    IMAGE_CHANNEL_DIM, MASK_VALID_FLOAT, MASK_VALID_UBYTE
from utils import read_points

#%%
class FaceSwap(object):
    """
    face swap
        - clone mode: cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER
    attributes:
        points (pd.dataframe) - columns: "x", "y"
        delaunay_triangulation (pd.dataframe)
            - columns: "VERT1", "VERT2", "VERT3"
            - [[tri_pt_idx, tri_pt_idx, tri_pt_idx],
               [tri_pt_idx, tri_pt_idx, tri_pt_idx],
               ...,
               [tri_pt_idx, tri_pt_idx, tri_pt_idx]]
        contour (1-dim array)
            - [pt_idx, pt_idx, ...,pt_idx]
    """

    _seamless_mode = [
        cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER
    ]

    #----------------------------------------------------------------------
    def __init__(self, points=None):
        """
        Constructor
        """
        if isinstance(points, pd.DataFrame):
            self._points = points
            self._delaunay_triangulation()
            self._convex_hull()
        elif isinstance(points, str):
            self._points = read_points(points)
            self._delaunay_triangulation()
            self._convex_hull()
        else:
            self._points = None
            self._triangulation_pt_indices = None
            self._contour_pt_indices = None

    def delaunay_affine_transform(self, image_src, image_dst,
                                  points_src, points_dst):
        """
        """
        image_dst_copy = image_dst.copy().astype(np.float32)

        # traversing triangules
        for _, tri_pt_idcs in self._triangulation_pt_indices.iterrows():

            triangule_src = pd.DataFrame(columns=["x", "y"], dtype=np.int)
            triangule_dst = pd.DataFrame(columns=["x", "y"], dtype=np.int)
            for pt_idx in tri_pt_idcs.values:
                triangule_src = triangule_src.append(
                    points_src.loc[pt_idx, :], ignore_index=True
                )
                triangule_dst = triangule_dst.append(
                    points_dst.loc[pt_idx, :], ignore_index=True
                )

            image_dst_copy = self._warp_affine_triangule(
                image_src=image_src, image_dst=image_dst_copy,
                triangule_src=triangule_src, triangule_dst=triangule_dst
            )

        # clamp
        image_dst_copy = np.clip(a=image_dst_copy, a_min=0, a_max=255)
        image_dst_copy = image_dst_copy.astype(np.uint8)

        return image_dst_copy

    def _delaunay_triangulation(self):
        """
        Delaunay triangulation
            https://blog.csdn.net/zhaoyin214/article/details/87942919
        """
        # rect (tuple) - "xmin", "ymin", "w", "h"
        rect = cv2.boundingRect(self._points.values)

        subdiv = cv2.Subdiv2D(rect)

        for _, pt in self._points.iterrows():
            subdiv.insert(tuple(pt))

        # index of points consist of a trangular
        triangulation_pt_indices = []

        triangule_list = subdiv.getTriangleList()

        for tri in triangule_list:

            # vertices of a triangule
            tri_vertices = [
                (tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])
            ]

            # index of points consist of a trangular
            tri_pt_idx_list = []
            for vertex in tri_vertices:
                # traverse all points
                for pt_idx, pt in self._points.iterrows():

                    if (abs(vertex[0] - pt.x) < 1) and \
                        (abs(vertex[1] - pt.y) < 1):
                        tri_pt_idx_list.append(pt_idx)

            triangulation_pt_indices.append(tri_pt_idx_list)

        self._triangulation_pt_indices = pd.DataFrame(
            triangulation_pt_indices, columns=TRIANGULATION_COLUMES
        )

    def _convex_hull(self):
        """
        convex hull of points
        """
        self._contour_pt_indices = cv2.convexHull(
            points=self._points.values, returnPoints=False
        )
        self._contour_pt_indices = self._contour_pt_indices.flatten()

    def _warp_affine_triangule(self, image_src, image_dst,
                               triangule_src, triangule_dst, alpha=1):
        """
        affine transform for each triangule

        arguments:
            image_src (2-dim array) - source
            image_dst (2-dim array) - destination
            triangule_src (pd.dataframe) - vertices of a triangule
                - columns: "x", "y"
            triangule_dst (pd.dataframe) - vertices of a triangule
                 - columns: "x", "y"

        return:
            image_copy (2-dim array), warpped image
        """
        rect_src = cv2.boundingRect(triangule_src.values)
        rect_dst = cv2.boundingRect(triangule_dst.values)
        # roi - "ymin", "xmin", "ymax", "xmax"
        roi_src = [
            rect_src[CV_RECT_YMIN],
            rect_src[CV_RECT_XMIN],
            rect_src[CV_RECT_YMIN] + rect_src[CV_RECT_HEIGHT],
            rect_src[CV_RECT_XMIN] + rect_src[CV_RECT_WIDTH]
        ]
        roi_dst = [
            rect_dst[CV_RECT_YMIN],
            rect_dst[CV_RECT_XMIN],
            rect_dst[CV_RECT_YMIN] + rect_dst[CV_RECT_HEIGHT],
            rect_dst[CV_RECT_XMIN] + rect_dst[CV_RECT_WIDTH]
        ]
        # offsets to roi origin
        triangule_src.x -= rect_src[CV_RECT_XMIN]
        triangule_src.y -= rect_src[CV_RECT_YMIN]
        triangule_dst.x -= rect_dst[CV_RECT_XMIN]
        triangule_dst.y -= rect_dst[CV_RECT_YMIN]

        image_src_roi = image_src[
            roi_src[0] : roi_src[2], roi_src[1] : roi_src[3], ...
        ]

        # affine matrix
        affine_matrix = cv2.getAffineTransform(
            src=triangule_src.values.astype(np.float32),
            dst=triangule_dst.values.astype(np.float32)
        )
        # affine transforming the source image
        # size w x h
        size = (rect_dst[CV_RECT_WIDTH], rect_dst[CV_RECT_HEIGHT])
        image_src_roi = cv2.warpAffine(src=image_src_roi,
                                       M=affine_matrix,
                                       dsize=size,
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)

        # mask of the destination triangule
        mask = np.zeros(shape=(rect_dst[CV_RECT_HEIGHT],
                               rect_dst[CV_RECT_WIDTH],
                               image_src.shape[IMAGE_CHANNEL_DIM]),
                        dtype=np.float32)
        cv2.fillConvexPoly(
            img=mask, points=triangule_dst.values, color=MASK_VALID_FLOAT
        )
        mask *= alpha

        image_dst[roi_dst[0] : roi_dst[2], roi_dst[1] : roi_dst[3]] = \
            mask * image_src_roi + \
            (1 - mask) * image_dst[roi_dst[0] : roi_dst[2], roi_dst[1] : roi_dst[3]]

        return image_dst

    def clone(self, image_src, image_dst, points_dst,
               mode=cv2.NORMAL_CLONE):
        """
        Poisson Image Editing
            P. Rez, M. Gangnet, A. Blake
            Acm Transactions on Graphics (2003)

            http://www.irisa.fr/vista/Papers/2003_siggraph_perez.pdf
            https://blog.csdn.net/zhaoyin214/article/details/88196575

        arguments:

        return:

        """
        contour_points = pd.DataFrame(columns=["x", "y"], dtype=np.int)
        for pt_idx in self._contour_pt_indices:
            contour_points = contour_points.append(
                points_dst.loc[pt_idx, :], ignore_index=True
            )

        mask = np.zeros(shape=image_dst.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            img=mask, points=contour_points.values, color=MASK_VALID_UBYTE
        )

        rect = cv2.boundingRect(points_dst.values)
        center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)

        if mode in self._seamless_mode:
            image_clone = cv2.seamlessClone(src=image_src,
                                            dst=image_dst,
                                            mask=mask,
                                            p=center,
                                            flags=mode)
        elif mode == "local_color_change":
            # image_clone = cv2.colorChange(src,
            #                               mask,
            #                               result,
            #                               red_mul,
            #                               green_mul,
            #                               blue_mul)
            pass
        elif mode == "local_illumination_change":
            # image_clone = cv2.illuminationChange(src,
            #                                      mask,
            #                                      result,
            #                                      alpha=0.2f,
            #                                      beta=0.4f)
            pass
        elif mode == "texture_flatten":
            # image_clone = cv2.textureFlattening(src,
            #                                     mask,
            #                                     dst,
            #                                     low_threshold,
            #                                     high_threshold,
            #                                     kernel_size)
            pass
        else:
            raise AssertionError("ERROR: invalid clone mode!")

        return image_clone

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if isinstance(points, pd.DataFrame):
            self._points = points
            self._delaunay_triangulation()
            self._convex_hull()
        elif isinstance(points, str):
            self._points = read_points(points)
            self._delaunay_triangulation()
            self._convex_hull()
        else:
            raise AssertionError(
                "ERROR: points {}, is not available!".format(type(points))
            )

    @property
    def delaunay_triangulation(self):
        return self._triangulation_pt_indices

    @property
    def contour(self):
        return self._contour_pt_indices


#%%
if __name__ == "__main__":

    from configs import FACE_LANDMARKS_68_PATH, PREDICTOR_68_PATH
    from configs import TEMPLATE_LANDMARKS_PATH
    from faceswap import FaceLocalization
    from utils import plot_images

    image_src = cv2.imread("./template/ip_man_3.jpg")
    image_dst = cv2.imread("./template/ronald-regan.jpg")

    landmarks = read_points(TEMPLATE_LANDMARKS_PATH)
    # image = cv2.imread(TEMPLATE_IMAGE_PATH)
    # rect = (0, 0, image.shape[1], image.shape[0])

    face_localizer = FaceLocalization(PREDICTOR_68_PATH,
                                      FACE_LANDMARKS_68_PATH)
    landmarks_src = face_localizer.get_face_landmarks(image_src)
    landmarks_dst = face_localizer.get_face_landmarks(image_dst)

    face_swap = FaceSwap(points=landmarks)
    print(face_swap.points)
    print(face_swap.delaunay_triangulation)
    print(face_swap.contour)

    image_face_wrap = face_swap.delaunay_affine_transform(
        image_src, image_dst, landmarks_src, landmarks_dst
    )
    image_face_clone = face_swap.clone(
        image_face_wrap, image_dst, landmarks_dst
    )

    plot_images([image_face_wrap, image_face_clone])

    cv2.imshow("face wrapped", image_face_wrap)
    cv2.imshow("face cloned", image_face_clone)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


#%%

