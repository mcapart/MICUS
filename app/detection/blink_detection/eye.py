import math
import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """
    # The first point is the one on the left, then 2 points for the top, the right and 2 points for the bottom

    FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                                   (374, 380), (380, 381), (381, 382), (382, 362),
                                   (263, 466), (466, 388), (388, 387), (387, 386),
                                   (386, 385), (385, 384), (384, 398), (398, 362)])

    FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                    (145, 153), (153, 154), (154, 155), (155, 133),
                                    (33, 246), (246, 161), (161, 160), (160, 159),
                                    (159, 158), (158, 157), (157, 173), (173, 133)])
    # 33 TO 133 IS THE TOP
    mediapipe_left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # 362 to 263 it the top
    mediapipe_right_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, ]

    def __init__(self, landmarks, side):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.width = None
        self.height = None
        self.EAR = None

        self._analyze(landmarks, side)


    def calculate_enhanced_ear(self, landmarks, eye_indices):
        """
        Calculate an enhanced Eye Aspect Ratio (EAR) for the given eye landmarks.

        Arguments:
        - landmarks: List of landmark points detected by Mediapipe.
        - eye_indices: List of indices for the eye landmarks.

        Returns:
        - ear: The calculated Eye Aspect Ratio.
        """


        eye_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in eye_indices])

        # Calculate the distances between the horizontal and vertical eye landmarks
        A = dist.euclidean(eye_points[1], eye_points[15])
        B = dist.euclidean(eye_points[2], eye_points[14])
        C = dist.euclidean(eye_points[3], eye_points[13])
        D = dist.euclidean(eye_points[4], eye_points[12])
        E = dist.euclidean(eye_points[5], eye_points[11])
        F = dist.euclidean(eye_points[6], eye_points[10])
        G = dist.euclidean(eye_points[7], eye_points[9])
        horizontal_dist = dist.euclidean(eye_points[0], eye_points[8])

        # Calculate the average of the vertical distances
        avg_vertical = (A + B + C + D + E + F + G) / 7.0

        # Calculate the Eye Aspect Ratio
        ear = avg_vertical / horizontal_dist
        return ear
   
    # def calculate_eye_size(self, landmarks, eye_indices):


    def _analyze(self, landmarks, side):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.mediapipe_left_eye
            #TODO i think the mediapipe landmarks are inverted
        elif side == 1:
            points = self.mediapipe_right_eye
        else:
            return
        if landmarks is not None:
            self.ear = self.calculate_enhanced_ear(landmarks, points)
        else:
            self.mediapipe_ear = -1


