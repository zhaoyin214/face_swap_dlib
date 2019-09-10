#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   user.py
@time    :   2019/09/09 10:57:21
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

#%%
import cv2
from faceswap import FaceLocalization, FaceSwap

#%%
class UserFaceSwapAPI(object):
    """
    """

    def __init__(self, user_id, image,
                 predictor_filepath,
                 face_landmarks_filepath,
                 clone_mode=cv2.NORMAL_CLONE):
        """
        """
        self._user_id = user_id
        self._face_localizer = FaceLocalization(
            predictor_filepath=predictor_filepath,
            face_landmarks_filepath=face_landmarks_filepath
        )
        self._image = image
        self._landmarks = self._face_localizer.get_face_landmarks(image)
        self._face_swap = FaceSwap(self._landmarks)
        self._clone_mode = clone_mode

    def image_clone(self, image_dst):

        landmarks_dst = self._face_localizer.get_face_landmarks(image_dst)
        if landmarks_dst is None:
            return image_dst

        image_copy = self._face_swap.delaunay_affine_transform(
            image_src=self._image, image_dst=image_dst,
            points_src=self._landmarks, points_dst=landmarks_dst
        )
        image_copy = self._face_swap.clone(
            image_src=image_copy, image_dst=image_dst,
            points_dst=landmarks_dst, mode=self._clone_mode
        )

        return image_copy

    def video_clone(self, video_filepath, video_output_filepath=None, is_show=False):

        video_cap = cv2.VideoCapture(video_filepath)
        frame_width= int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        if video_output_filepath is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(
                video_output_filepath, fourcc, fps,
                (frame_width, frame_height)
            )
        else:
            video_writer = None

        while True:

            is_grabbed, frame = video_cap.read()

            if not is_grabbed:
                break

            # clone
            frame = self.image_clone(image_dst=frame)

            if video_writer is not None:
                video_writer.write(frame)

            if is_show:
                cv2.imshow("face clone", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video_cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    @property
    def user_face(self):
        return self._image

    @user_face.setter
    def user_face(self, image):
        self._image = image
        self._landmarks = self._face_localizer.get_face_landmarks(image)
        self._face_swap = FaceSwap(self._landmarks)

    @property
    def clone_mode(self):
        return self._clone_mode

    @clone_mode.setter
    def clone_mode(self, clone_mode):
        self._clone_mode = clone_mode


#%%
if __name__ == "__main__":

    from configs import FACE_LANDMARKS_68_PATH, PREDICTOR_68_PATH

    image_user = cv2.imread("./template/george-w-bush.jpg")
    image_dst = cv2.imread("./template/ronald-regan.jpg")

    user_face_swap = UserFaceSwapAPI(
        user_id=0, image=image_user,
        predictor_filepath=PREDICTOR_68_PATH,
        face_landmarks_filepath=FACE_LANDMARKS_68_PATH
    )

    image_clone = user_face_swap.image_clone(image_dst=image_dst)
    cv2.imshow("image clone", image_clone)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    video_filepath = "./template/744f0638ee69e4881b4f12fe47c93962.mp4"
    video_output_filepath = "./output/test.avi"
    user_face_swap.video_clone(
        video_filepath=video_filepath,
        video_output_filepath=video_output_filepath,
        is_show=True
    )

#%%
