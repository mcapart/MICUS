import math
import numpy as np
import cv2
from gaze_tracking.pupil import Pupil
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
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

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

    def __init__(self, original_frame, dlib_landmarks, media_pipe_landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.width = None
        self.height = None
        self.EAR = None

        self._analyze(original_frame, dlib_landmarks, media_pipe_landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))
        self.width = eye_width
        self.height = eye_height

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None
        return ratio

    def _EAR(self, landmarks, points):
        eye_points = {}
        for idx, p in enumerate(points):
            eye_points[idx] = (landmarks.part(p).x, landmarks.part(p).y)

        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye_points[0], eye_points[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear

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
    def _analyze(self, original_frame, dlib_landmarks: dlib.full_object_detection, media_pipe_landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            dlib_points = self.LEFT_EYE_POINTS
            mediapipe_points = self.mediapipe_left_eye
            #TODO i think the mediapipe landmarks are inverted
        elif side == 1:
            dlib_points = self.RIGHT_EYE_POINTS
            mediapipe_points = self.mediapipe_right_eye
        else:
            return

        self.blinking = self._blinking_ratio(dlib_landmarks, dlib_points)
        self.EAR = self._EAR(dlib_landmarks, dlib_points)
        if media_pipe_landmarks is not None:
            self.mediapipe_ear = self.calculate_enhanced_ear(media_pipe_landmarks, mediapipe_points)
        else:
            self.mediapipe_ear = -1
        #print(self.mediapipe_ear, self.EAR)

        #print("blinking ", self.blinking, " ear ", self.EAR)
        self._isolate(original_frame, dlib_landmarks, dlib_points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
