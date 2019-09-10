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

#%% ---
import cv2
import dlib
import numpy as np
import pandas as pd
import json
import os

#%% ---
class FaceLocalization(object):
    """
    Face localization with dlib, face detection and face alignment
        - face detection, detect_faces
        - face alignment, get_face_landmarks, draw_face_landmarks
    """

    def __init__(self,
                 predictor_filepath,
                 face_landmarks_filepath):
        """
        Constructor
        """

        self._detector = dlib.get_frontal_face_detector()
        assert os.path.isfile(predictor_filepath), \
            "ERROR: {} does not exist!".format(predictor_filepath)
        self._shape_predictor = dlib.shape_predictor(predictor_filepath)
        assert os.path.isfile(face_landmarks_filepath), \
            "ERROR: {} does not exist!".format(face_landmarks_filepath)
        self._load_face_landmarks(face_landmarks_filepath)

    def _load_face_landmarks(self, filepath):

        with open(filepath, "r") as fp:
            self._face_landmarks_config = json.load(fp)

    def get_face_landmarks(self, image):
        """
        arguments:
            image (2-dim array)

        return:
            landmarks(pd.dataframe) - columns: "x", "y"
        """

        rects = self._detector(image, 1)

        # if len(rects) > 1:
        #     raise AssertionError("too many faces")
        if len(rects) == 0:
            return None

        landmarks = np.asarray([
            [pt.x, pt.y] for pt in \
            self._shape_predictor(image, rects[0]).parts()
        ])
        landmarks = pd.DataFrame(landmarks, columns=["x", "y"])

        return landmarks

    def detect_faces(self, image):
        """
        arguments:
            image (2-dim array)

        return:
            bounding_boxes (dlib.rectangules)
                - top(), bottom(), left(), right()
        """

        bounding_boxes = self._detector(image, 1)

        return bounding_boxes

    def draw_face_bboxes(self, image, bounding_boxes):
        """
        arguments:
            image (2-dim array)
            bounding_boxes (dlib.rectangules)
                - top(), bottom(), left(), right()

        return:
            image_copy (2-dim array)
        """
        image_copy = image.copy()

        for bbox in bounding_boxes:

            cv2.rectangle(img=image_copy,
                          pt1=(bbox.left(), bbox.top()),
                          pt2=(bbox.right(), bbox.bottom()),
                          color=(255, 255, 255), thickness=1)

        return image_copy

    def draw_face_landmarks(self, image, landmarks):
        """
        arguments:
            image (2-dim array)
            landmarks(pd.dataframe) - columns: "x", "y"

        return:
            image_copy (2-dim array)
        """
        image_copy = image.copy()

        for idx, point in landmarks.iterrows():
            pos = (point.x, point.y)
            # cv2.putText(image_copy, str(idx), pos,
            #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            #             fontScale=0.4,
            #             color=(0, 0, 255))
            cv2.circle(image_copy, pos, 3, color=(0, 255, 255))

        return image_copy

    @property
    def face_landmarks_config(self):
        return self._face_landmarks_config


#%%
if __name__ == "__main__":

    from configs import FACE_LANDMARKS_68_PATH, PREDICTOR_68_PATH

    image = cv2.imread("./template/ip_man_3.jpg")

    face_localizer = FaceLocalization(PREDICTOR_68_PATH,
                                      FACE_LANDMARKS_68_PATH)
    print(face_localizer.face_landmarks_config)
    rects = face_localizer.detect_faces(image)
    landmarks = face_localizer.get_face_landmarks(image)

    cv2.imshow("bounding boxes",
               face_localizer.draw_face_bboxes(image, rects))
    cv2.imshow("landmarks",
               face_localizer.draw_face_landmarks(image, landmarks))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

#%%
