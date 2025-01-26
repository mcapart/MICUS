from __future__ import division
import os
import cv2
import dlib
from .eye import Eye


# Code changed modified from https://github.com/antoinelame/GazeTracking
class BlinkTracking(object):
    """
    This class tracks the user's blinking.
    """

    def __init__(self):
        self.eye_left: Eye = None
        self.eye_right: Eye = None
        self.landmarks = None
        self.faces = []
        self.no_landmark = 0


        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "../../models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)


    def analyze(self, landmarks ,frame):
        """Initialize Eye objects

        Arguments:
            landmarks
            frame
        """
        self.landmarks = landmarks
        if landmarks is None:
            self.no_landmark += 1
        try:
            self.eye_left = Eye( landmarks, 0)
            self.eye_right = Eye( landmarks, 1)

        except IndexError:
            self.eye_left = None
            self.eye_right = None




