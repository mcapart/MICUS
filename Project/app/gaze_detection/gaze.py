import cv2
import numpy as np
from app.blink_detection.helpers import relative, relativeT

class Gaze:
    def __init__(self, frame, points):
        self.frame = frame
        self.points = points
        self.left_gaze_line = None
        self.right_gaze_line = None
        self.calculate_gaze()

    def calculate_gaze(self):
        image_points = np.array([
            relative(self.points.landmark[4], self.frame.shape),
            relative(self.points.landmark[152], self.frame.shape),
            relative(self.points.landmark[263], self.frame.shape),
            relative(self.points.landmark[33], self.frame.shape),
            relative(self.points.landmark[287], self.frame.shape),
            relative(self.points.landmark[57], self.frame.shape)
        ], dtype="double")

        image_points1 = np.array([
            relativeT(self.points.landmark[4], self.frame.shape),
            relativeT(self.points.landmark[152], self.frame.shape),
            relativeT(self.points.landmark[263], self.frame.shape),
            relativeT(self.points.landmark[33], self.frame.shape),
            relativeT(self.points.landmark[287], self.frame.shape),
            relativeT(self.points.landmark[57], self.frame.shape)
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0, -63.6, -12.5),
            (-43.3, 32.7, -26),
            (43.3, 32.7, -26),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])

        focal_length = self.frame.shape[1]
        center = (self.frame.shape[1] / 2, self.frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        left_pupil = relative(self.points.landmark[468], self.frame.shape)
        right_pupil = relative(self.points.landmark[473], self.frame.shape)

        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)

        if transformation is not None:
            self.left_gaze_line = self.calculate_eye_gaze(left_pupil, Eye_ball_center_left, transformation,
                                                          rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            self.right_gaze_line = self.calculate_eye_gaze(right_pupil, Eye_ball_center_right, transformation,
                                                           rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    def calculate_eye_gaze(self, pupil, eye_ball_center, transformation, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
        pupil_world_cord = transformation @ np.array([[pupil[0], pupil[1], 0, 1]]).T
        S = eye_ball_center + (pupil_world_cord - eye_ball_center) * 10

        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        gaze = pupil + (eye_pupil2D[0][0] - pupil) - (head_pose[0][0] - pupil)

        p1 = (int(pupil[0]), int(pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        return (p1, p2)

    def get_gaze_lines(self):
        return self.left_gaze_line, self.right_gaze_line
