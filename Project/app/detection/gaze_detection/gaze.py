import cv2
import numpy as np
from app.detection.blink_detection.helpers import (relative, relativeT)

DISTANCE_FROM_EYE_BALL_CENTER = 10

class Gaze:
    def __init__(self, frame, points):
        self.frame = frame
        self.points = points
        self.left_gaze_line = None
        self.right_gaze_line = None
        self.calculate_gaze()

    def calculate_gaze(self):
        """
        The gaze function gets an image and face landmarks from mediapipe framework.
        The function draws the gaze direction into the frame.
        """

        '''
        2D image points.
        relative takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y) format
        '''
        image_points = np.array([
            relative(self.points.landmark[4], self.frame.shape),  # Nose tip
            relative(self.points.landmark[152], self.frame.shape),  # Chin
            relative(self.points.landmark[263], self.frame.shape),  # Left eye left corner
            relative(self.points.landmark[33], self.frame.shape),  # Right eye right corner
            relative(self.points.landmark[287], self.frame.shape),  # Left Mouth corner
            relative(self.points.landmark[57], self.frame.shape)  # Right mouth corner
        ], dtype="double")

        image_points1 = np.array([
            relativeT(self.points.landmark[4], self.frame.shape),
            relativeT(self.points.landmark[152], self.frame.shape),
            relativeT(self.points.landmark[263], self.frame.shape),
            relativeT(self.points.landmark[33], self.frame.shape),
            relativeT(self.points.landmark[287], self.frame.shape),
            relativeT(self.points.landmark[57], self.frame.shape)
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])
        '''
        3D model eye points
        The center of the eye ball
        '''
        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])
        
        '''
        camera matrix estimation
        '''
        focal_length = self.frame.shape[1] 
        center = (self.frame.shape[1] / 2, self.frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # 2d pupil location
        left_pupil = relative(self.points.landmark[468], self.frame.shape)
        right_pupil = relative(self.points.landmark[473], self.frame.shape)
        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)

        if transformation is not None: 
            self.left_gaze_line = self.calculate_eye_gaze(left_pupil, Eye_ball_center_left, transformation,
                                                          rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            self.right_gaze_line = self.calculate_eye_gaze(right_pupil, Eye_ball_center_right, transformation,
                                                           rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    def calculate_eye_gaze(self, pupil, eye_ball_center, transformation, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
        # project pupil image point into 3d world point 
        pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T
        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S = eye_ball_center + (pupil_world_cord - eye_ball_center) * DISTANCE_FROM_EYE_BALL_CENTER
        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                 translation_vector, camera_matrix, dist_coeffs)
        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector, translation_vector, camera_matrix, dist_coeffs)
         # correct gaze for head rotation
        gaze = pupil + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)

        p1 = (int(pupil[0]), int(pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        return (p1, p2)

    def get_gaze_lines(self):
        return self.left_gaze_line, self.right_gaze_line
